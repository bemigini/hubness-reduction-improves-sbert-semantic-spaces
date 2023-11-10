# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:39:17 2022

@author: bmgi



Using Sentence BERT



"""


from collections.abc import Callable
import csv
from datetime import datetime

import gzip
import logging

import math
import numpy as np
from numpy.typing import NDArray
import os
import pandas as pd

from src.datasets import loading_datasets as loadd
from src.modules import CenterEmbeddings, DistanceLoss, EmbeddingDistanceEvaluator, ZScoreNorm
from src.util import sent_dist_util as sdu
from src.util import random_util, vectors, naming, save_to_file
from src.util.train_parameters import load_train_parameters, TrainParameters 

from sentence_transformers import SentenceTransformer, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction, TripletEvaluator
from sentence_transformers.readers import InputExample

from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from typing import Dict, Iterable, Tuple



# Pretrained models

# Sent embedding performance 69.57, semantic search 57.02
# By default, input text longer than 384 word pieces is truncated.
all_mpnet_base = 'all-mpnet-base-v2' 

# Sent embedding performance 65.98, semantic search 52.83
multi_qa_distilbert_base = 'multi-qa-distilbert-cos-v1' 

# Sent embedding performance 68.73, semantic search 50.94
all_distilroberta = 'all-distilroberta-v1'

# Sent embedding performance 68.70, semantic search 50.82
all_minilm = 'sentence-transformers/all-MiniLM-L12-v2' 


def get_pretrained_sentence_bert_model_names():
    return [all_mpnet_base, multi_qa_distilbert_base, all_distilroberta, all_minilm]


def get_embeddings_h5_dataset_name():
    return 'embeddings'


def get_sentence_bert_embeddings(
        sentences: Iterable[str], 
        model_name: str,
        device: str = 'cpu') -> NDArray:
    model = SentenceTransformer(model_name, device = device)
    return model.encode(sentences)


def get_and_save_sentence_bert_embeddings(
        sentences: Iterable[str], 
        model_name: str,
        save_to: str,
        device: str = 'cpu') -> NDArray:
    embeddings = get_sentence_bert_embeddings(sentences, model_name, device)
    
    # Save embeddings in hdf5 gzipped format
    save_to = save_to.replace('.csv', '').replace('.gz', '')
    if not save_to.endswith('.h5'):
        save_to = save_to + '.h5'
    h5_dataset_name = get_embeddings_h5_dataset_name()
    save_to_file.save_to_hdf5(embeddings, save_to, h5_dataset_name)
    
    return embeddings


def get_and_save_sentence_bert_embeddings_from_model_folder(
        output_folder: str,
        model_prefix: str,
        dataset: str,
        subset: str,
        sentences: Iterable[str],
        device: str) -> None:
    
    model_names = naming.get_model_names_from_model_folder(output_folder, model_prefix)
    
    embedding_folder_path = os.path.join(output_folder, naming.get_embedding_folder())
    os.makedirs(embedding_folder_path, exist_ok = True)
    
    for model_name in tqdm(model_names):
        embedding_file_name = naming.get_model_embedding_name(model_name, dataset, subset)
        embedding_save_path = os.path.join(output_folder, naming.get_embedding_folder(), embedding_file_name)
        
        if os.path.exists(embedding_save_path):
            continue
        
        model_load_path = os.path.join(output_folder, naming.get_model_folder(), model_name)
        
        get_and_save_sentence_bert_embeddings(
                sentences = sentences, 
                model_name = model_load_path,
                save_to = embedding_save_path,
                device = device)


def load_sentence_bert_embeddings(load_from: str) -> NDArray:  
    dataset_name = get_embeddings_h5_dataset_name()
    
    embeddings = save_to_file.load_from_hdf5(load_from, dataset_name)
    
    return embeddings


def load_subset_embeddings(
        model_name: str, 
        dataset: str, 
        subset: str,
        embedding_folder_path: str):
    emb_name = naming.get_model_embedding_name(model_name, dataset, subset)
    
    model_embedding_path = os.path.join(embedding_folder_path, emb_name)
    
    if os.path.exists(model_embedding_path):
        embeddings = load_sentence_bert_embeddings(model_embedding_path)
    else:
        raise ValueError(f'Path did not exist: {model_embedding_path}')
        
    return embeddings


def get_sent_id_to_embedding(
        sentence_df: pd.DataFrame, 
        id_col: str, 
        sent_col: str, 
        model_name: str) -> Dict[int, NDArray]:
    sentences = sentence_df[sent_col].values
    embeddings = get_sentence_bert_embeddings(sentences, model_name)
    sent_id_to_emb = {sent_id: embeddings[i] 
                      for i, sent_id 
                      in enumerate(sentence_df[id_col].values) }
    
    return sent_id_to_emb


def add_distance_col_to_df(
        pair_df: pd.DataFrame,
        sent1_id_col: str,
        sent2_id_col: str,
        sent_id_to_emb: Dict[int, NDArray],
        dist_metric: Callable[[NDArray, NDArray], float]) -> pd.DataFrame:
    
    sent1_emb = [sent_id_to_emb[i] for i in pair_df[sent1_id_col].values]
    sent2_emb = [sent_id_to_emb[i] for i in pair_df[sent2_id_col].values]
    
    distances = [dist_metric(sent1_emb[i], sent2_emb[i]) 
                 for i in range(len(sent1_emb))]
    
    pair_df.insert(pair_df.shape[1], 'measured_dist', value = distances)
    
    return pair_df
    
    
def get_sts_sent_bert_distances(
        sentence_dist_folder: str,
        model_name: str,
        remove_stop_words: bool,
        dist_metric: Callable[[NDArray, NDArray], float]) -> pd.DataFrame:
    
    sentence_df = loadd.load_sts_sentence_id_csv(sentence_dist_folder)
    id_col = 'id'
    sent_col = 'sentence'
    
    if remove_stop_words:
        sent_no_stop_col = 'sentence_no_stop'
        remove_stop_vec = np.vectorize(sdu.remove_stopwords)
        sentence_df.loc[:, sent_no_stop_col] = remove_stop_vec(sentence_df[sent_col].values)
        sent_col = sent_no_stop_col
        
    
    print('Getting embeddings')
    sent_id_to_emb = get_sent_id_to_embedding(sentence_df, id_col, sent_col, model_name)
    
    sts_df = loadd.load_sts_benchmark_combined_with_sent_ids(sentence_dist_folder)    
    sent1_id_col = 'sentence1_id'
    sent2_id_col = 'sentence2_id'
    
    print('Getting distances')
    sts_df = add_distance_col_to_df(sts_df, sent1_id_col, sent2_id_col, sent_id_to_emb, dist_metric)
    
    return sts_df


def get_sentence_bert_sts_data(
        distance_function: str,
        max_dist_relation: vectors.VectorRelation) -> Tuple[Iterable[InputExample], Iterable[InputExample], Iterable[InputExample]]:
    #Check if dataset exsist. If not, download and extract  it
    sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)
    
    train_samples = []
    dev_samples = []
    test_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if max_dist_relation == vectors.VectorRelation.ORTHOGONAL:
                normed_score = float(row['score']) / 5.0  # Normalize score to range 0 to 1
            elif max_dist_relation == vectors.VectorRelation.OPPOSITE:
                normed_score = float(row['score']) / 2.5  # Normalize score to range 0 to 2
            else:
                raise NotImplementedError('Unknown maximum distance relation')
                
            if distance_function == 'cos':
                # We get a similarity between 1 and 0 (orthogonal) 
                # or between 2 and 0 (opposite)
                score = normed_score 
                
            elif distance_function == 'cos_dist':
                if max_dist_relation == vectors.VectorRelation.ORTHOGONAL:
                    # We get a distance between 0 and 1
                    score = 1 - normed_score 
                else:
                    # We get a distance between 0 and 2
                    score = 2 - normed_score 
                    
            elif distance_function == 'euclidean':
                if max_dist_relation == vectors.VectorRelation.ORTHOGONAL:
                    # Normalize score to range -1 to 1
                    normed_score = (float(row['score']) - 2.5) / 2.5  
                    # We get a distance between 0 and sqrt(2)
                    score = math.sqrt(1 - normed_score) 
                else:
                    # Normalize score to range 0 to 4
                    normed_score = float(row['score']) / (5./4.)  
                    # We get a distance between 0 and 2
                    # With a normalized score of 2 corresponding to a distance of sqrt(2) 
                    score = math.sqrt(4 - normed_score) 
                    
            else:
                raise NotImplementedError(f'Distance not implemented: {distance_function}')
                
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

            if row['split'] == 'dev':
                dev_samples.append(inp_example)
            elif row['split'] == 'test':
                test_samples.append(inp_example)
            else:
                train_samples.append(inp_example)
                
    return train_samples, dev_samples, test_samples    


# output_embedding_size = 0 will use the same embedding size as for the words
def build_model_architecture(
        model_name: str,
        max_seq_length: int, 
        output_embedding_size: int,
        center_embeddings: bool,
        normalize_embeddings: bool,
        z_score_norm: bool,
        device: str
        ) -> SentenceTransformer:
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(
        model_name, 
        max_seq_length = max_seq_length)
    
    word_embedding_dim = word_embedding_model.get_word_embedding_dimension()
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_dim,
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    
    modules = [word_embedding_model, pooling_model]
    
    if (output_embedding_size > 0) and (output_embedding_size < word_embedding_dim):
        modules.append(models.Dense(
            in_features = word_embedding_dim,
            out_features = output_embedding_size,
            activation_function = nn.Tanh()))    
    
    if z_score_norm and center_embeddings:
        raise ValueError('It does not make sente to both center features and do z score normalization')
    
    if z_score_norm:
        modules.append(ZScoreNorm.ZScoreNorm())
        
    if center_embeddings:
        modules.append(CenterEmbeddings.CenterEmbeddings())
    
    if normalize_embeddings:
        modules.append(models.Normalize())

    model = SentenceTransformer(modules = modules, device = device)
    return model


def get_pair_loss_and_evaluator(
        model: SentenceTransformer,
        dev_samples: Iterable[InputExample],
        distance: str):
    
    if distance == 'euclidean':
        train_loss = DistanceLoss.DistanceLoss(
            model = model,
            dist = DistanceLoss.DistanceFunction.EUCLIDEAN)
        main_distance = SimilarityFunction.EUCLIDEAN
        
        evaluator = EmbeddingDistanceEvaluator.EmbeddingDistanceEvaluator.from_input_examples(
            dev_samples, 
            name='dev',
            main_distance = main_distance)
        
    elif distance == 'cos':        
        train_loss = losses.CosineSimilarityLoss(model=model)
        main_similarity = SimilarityFunction.COSINE
        
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            dev_samples, 
            name='dev',
            main_similarity = main_similarity)
        
    elif distance == 'cos_dist':        
        train_loss = DistanceLoss.DistanceLoss(
            model = model, 
            dist = DistanceLoss.DistanceFunction.COSINE)
        main_distance = SimilarityFunction.COSINE
        
        evaluator = EmbeddingDistanceEvaluator.EmbeddingDistanceEvaluator.from_input_examples(
            dev_samples, 
            name='dev',
            main_distance = main_distance)
        
    else:
        raise NotImplementedError(f'Distance not implemented: {distance}')
    
    
    
    return train_loss, evaluator


def get_triplet_loss_and_evaluator(
        model: SentenceTransformer,
        dev_samples: Iterable[InputExample],
        distance: str,
        triplet_margin: float):
    
    if distance == 'euclidean':
        loss_distance = losses.TripletDistanceMetric.EUCLIDEAN
        main_distance_function = SimilarityFunction.EUCLIDEAN
    elif distance == 'cos_dist':        
        loss_distance = losses.TripletDistanceMetric.COSINE
        main_distance_function = SimilarityFunction.COSINE
    else:
        raise NotImplementedError(f'Distance not implemented: {distance}')
    
    train_loss = losses.TripletLoss(
        model = model, 
        distance_metric = loss_distance,
        triplet_margin = triplet_margin)
    
    evaluator = TripletEvaluator.from_input_examples(
        dev_samples, 
        main_distance_function = main_distance_function, 
        name = 'dev')
    
    return train_loss, evaluator


def train_sentence_bert(
        train_samples: Iterable[InputExample],
        dev_samples: Iterable[InputExample],
        train_dataset: IterableDataset,
        dev_dataset: IterableDataset,
        model_save_to: str,
        dataset_name: str,
        device: str,
        normalize_embeddings: bool,
        center_embeddings: bool,
        z_score_norm: bool = False,
        loss_type: str = 'pair',
        triplet_margin: float = 0.5,
        distance: str = 'euclidean',
        pretrained_model_name: str = 'microsoft/mpnet-base',
        max_seq_length: int = 128,
        output_embedding_size: int = 0,
        train_batch_size: int = 16,
        num_epochs: int = 4) -> None:

    #You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
    # 'microsoft/mpnet-base' or 'distilbert-base-uncased'
    model_name = pretrained_model_name

    if not model_save_to or not model_save_to.strip():
        model_save_path = f'output/{dataset_name}_{model_name.replace("/", "-")}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    else:
        model_save_path = model_save_to

    model = build_model_architecture(
        model_name = model_name,
        max_seq_length = max_seq_length,
        output_embedding_size = output_embedding_size,
        center_embeddings = center_embeddings,
        normalize_embeddings = normalize_embeddings,
        z_score_norm = z_score_norm,
        device = device)
    
    
    if loss_type == 'pair':
        train_loss, evaluator = get_pair_loss_and_evaluator(
            model = model, 
            dev_samples = dev_samples,
            distance = distance)
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
        
    elif loss_type == 'triplet':
        dev_samples = [example for example in dev_dataset]
        train_loss, evaluator = get_triplet_loss_and_evaluator(
            model = model, 
            dev_samples = dev_samples,
            distance = distance,
            triplet_margin = triplet_margin)
        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=train_batch_size)
        

    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=model_save_path)


def train_sentence_bert_sts(
        device: str,
        model_save_to: str,
        normalize_embeddings: bool,
        center_embeddings: bool,
        z_score_norm: bool = False,
        distance: str = 'euclidean',
        max_dist_relation: vectors.VectorRelation = vectors.VectorRelation.ORTHOGONAL,
        pretrained_model_name: str = 'microsoft/mpnet-base',
        max_seq_length: int = 256,
        train_batch_size: int = 16,
        num_epochs: int = 4):
    
    train_samples, dev_samples, test_samples = get_sentence_bert_sts_data(
        distance,
        max_dist_relation = max_dist_relation)    
    
    train_sentence_bert(
            train_samples = train_samples,
            dev_samples = dev_samples,
            train_dataset = None,
            dev_dataset = None,
            model_save_to = model_save_to,
            dataset_name = 'sts_sentence_bert',
            device = device,
            normalize_embeddings = normalize_embeddings,
            center_embeddings = center_embeddings,
            z_score_norm = z_score_norm,
            loss_type = 'pair',
            distance = distance,
            pretrained_model_name = pretrained_model_name,
            max_seq_length = max_seq_length,
            train_batch_size = train_batch_size,
            num_epochs = num_epochs)


def train_sentence_bert_newsgroups(
        device: str,
        model_save_to: str,
        normalize_embeddings: bool,
        center_embeddings: bool,
        z_score_norm: bool = False,
        triplet_margin: float = 0.5,
        train_dataset_type: str = 'many',
        distance: str = 'euclidean',        
        pretrained_model_name: str = 'microsoft/mpnet-base',
        max_seq_length: int = 128, 
        output_embedding_size: int = 0,
        train_batch_size: int = 16,
        num_epochs: int = 4):
    
    train_dataset, dev_dataset = loadd.get_train_dev_TripletDataset_from_newsgroups(
        triplets_per_anchor = 4, 
        with_replacement = False,
        train_dataset_type = train_dataset_type)
    
    train_sentence_bert(
            train_samples = None,
            dev_samples = None,
            train_dataset = train_dataset,
            dev_dataset = dev_dataset,
            model_save_to = model_save_to,
            dataset_name = 'sts_sentence_bert',
            device = device,
            normalize_embeddings = normalize_embeddings,
            center_embeddings = center_embeddings,
            z_score_norm = z_score_norm,
            loss_type = 'triplet',
            triplet_margin = triplet_margin,
            distance = distance,
            pretrained_model_name = pretrained_model_name,
            max_seq_length = max_seq_length,
            output_embedding_size = output_embedding_size,
            train_batch_size = train_batch_size,
            num_epochs = num_epochs)


def train_models_with_parameters(
        seed: int, 
        device: str,
        output_folder: str,
        train_parameters: TrainParameters,
        train_batch_size: int = 16,
        num_epochs: int = 4) -> None:
    
    cuda = 'cuda' in device
    random_util.set_seed(seed, cuda)
    
    model_save_names = []
    
    for base_model in train_parameters.base_models:
        for distance in train_parameters.distances:
            for max_dist_relation in train_parameters.max_difference:
                for centering in train_parameters.centering_parameters:
                    
                    z_score_norm = centering.z_score_norm
                    normalize_embeddings = centering.normalize_embeddings                    
                    center_embeddings = centering.center_embeddings
                    
                    
                    model_name_no_slash = base_model.replace('/', '-').replace('\\', '-')
                    
                    model_save_name = naming.get_sts_bert_model_save_name(
                        model_name_no_slash,
                        distance,
                        max_dist_relation.name,
                        z_score_norm,
                        normalize_embeddings,
                        center_embeddings,
                        seed)
                    
                    model_save_to = os.path.join(output_folder, naming.get_model_folder(), model_save_name)
                    
                    model_save_names.append(model_save_name)
                    
                    if os.path.isdir(model_save_to):
                        continue
                    

                    train_sentence_bert_sts(
                        device = device,
                        model_save_to = model_save_to,
                        normalize_embeddings = normalize_embeddings, 
                        center_embeddings = center_embeddings,
                        z_score_norm = z_score_norm,
                        distance = distance,
                        max_dist_relation = max_dist_relation,
                        pretrained_model_name = base_model,
                        train_batch_size = train_batch_size,
                        num_epochs = num_epochs)

    
    model_name_file = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-model-names.csv'
    model_name_path = os.path.join(output_folder, model_name_file)    
    
    with open(model_name_path, 'w') as f:
        f.write(','.join(model_save_names))
    
    
def train_models_with_parameters_path(
        seed: int, 
        device: str,
        output_folder: str,
        train_parameters_path: str,
        train_batch_size: int = 16,
        num_epochs: int = 4) -> None:
    
    train_parameters = load_train_parameters(train_parameters_path)
    
    train_models_with_parameters(
            seed = seed, 
            device = device,
            output_folder = output_folder,
            train_parameters = train_parameters,
            train_batch_size = train_batch_size,
            num_epochs = num_epochs)



