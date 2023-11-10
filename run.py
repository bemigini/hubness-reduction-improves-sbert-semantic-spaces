# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:43:53 2022

@author: bmgi

run.py: Run Script for training sentence BERT model and getting embeddings and results


Usage:
    run.py train --output-folder=<file> --training-parameters=<file> [options]
    run.py train --output-folder=<file> --model-name=<string> --dataset=<string> [options]
    run.py embeddings --output-folder=<file> --model-name=<string> --dataset=<string> --embeddings-save-to=<file> [options]
    run.py embeddings --output-folder=<file> --model-name=<string> --dataset=<string> --model-names-path=<file> [options]
    run.py performance --output-folder=<file> --dataset=<string> --model-names-path=<file> [options]
    
Options:
    -h --help                               show this screen.
    --cuda                                  use GPU    
    --sentences=<file>                      sentences csv file. If not given will fall back to dataset.
    --dataset=<string>                      use known dataset dataset_name.subset. Dataset names are newsgroups, ag_news, yahoo_answers. Subsets are train, test, all [default: newsgroups.all]
    --seed=<int>                            seed [default: 0]
    --output-folder=<file>                  folder where models/embeddings/results are saved to/loaded from [default: /scratch/bmgi]
    --model-name=<string>                   model name to save to/load from [default: custom]
    --embeddings-save-to=<file>             embeddings save path [default: /scratch/bmgi/embeddings.csv]  
    --training-parameters=<file>            path to json file with training parameters. Train models with combinations of these parameters.
    --model-names-path=<file>               path to model names to test performance of. If 'embeddings' will test performance of all models found in embeddings folder. If 'models' will test performance of all models found in models folder
    --pretrained-model-name=<string>        pretrained transformer model for training [default: microsoft/mpnet-base]    
    --train-batch-size=<int>                batch size for training [default: 16]
    --epochs=<int>                          number of epochs to train [default: 4]    
    --normalize-embeddings                  normalize sentence bert embeddings
    --center-embeddings                     center the features of the sentence bert embeddings
    --z-score-normalization                 do z score normalization on sentence bert embeddings
    --distance=<string>                     distance function to use: euclidean, cos(cosine similarity) or cos_dist(cosine distance) [default: euclidean]
    --triplet-margin=<float>                margin to use for triplet loss [default: 0.5]
    --train-dataset-type=<string>           use sample of triplets for training 'sample', many combinations of triplets 'many' or all combinations of triplets 'all'. [default: many]
    --max-seq-length=<int>                  max sequence length to use in model [default: 128]
    --embedding-size=<int>                  size of the output embedding. 0 means same as the word embedding size of the word embedding model [default: 0]    
    --num-fold=<int>                        number of folds to use in d-fold validation. [default: 10]
    --center-norm-after-train               center dimensions and normalise embeddings after training is done.
    --force-dist=<string>                   transform dimensions of embeddings to be specific distribution before using KNN. 'normal_all', 'normal_splits', 'uniform_all', 'uniform_splits' or 'none'. 'all' transform using all embeddings, 'splits' transform within train/test splits, 'none' do not transform. [default: none]
    --skhubness-reduction=<string>          use an skhubness hub reduction method on KNN graphs. 'none', 'mutual_proximity' or 'local_scaling' [default: none]
    --result-type=<string>                  which results to get from performance command. 'knn' or 'hubness'. 'hubness' currently only works with --model-names-path=embeddings [default: knn]
    --n-jobs=<int>                          number of jobs to start on large datasets. [default: 5]

"""



from docopt import docopt

import logging

import os

from sentence_transformers import LoggingHandler
from src import sentence_bert as sent_bert
from src.datasets import loading_datasets as loadd
from src.performance import saving_results as savr
from src.hubness.reduction import ForceDistribution, SkhubnessReduction
from src.util import naming, random_util


from typing import Dict




def train(args: Dict):
    
    device = "cuda" if args['--cuda'] else "cpu"
    
    output_folder = args['--output-folder'] if args['--output-folder'] else ''
    train_batch_size = int(args['--train-batch-size'])
    num_epochs = int(args['--epochs'])
    
    train_parameters = args['--training-parameters']
    
    
    if train_parameters and train_parameters.strip():
        seed = int(args['--seed'])
        sent_bert.train_models_with_parameters_path(
            seed = seed, 
            device = device,
            output_folder = output_folder,
            train_parameters_path = train_parameters,
            train_batch_size = train_batch_size,
            num_epochs = num_epochs)
    
    elif args['--sentences']:        
        NotImplementedError('--sentences not implemented for training.')
    else:
        pretrained_model_name = args['--pretrained-model-name']
        model_name_save_to = args['--model-name'] if args['--model-name'] else ''
        dataset, subset = args['--dataset'].split('.')
        
        normalize_embeddings = True if args['--normalize-embeddings'] else False
        center_embeddings = True if args['--center-embeddings'] else False    
        z_score_norm = True if args['--z-score-normalization'] else False
        
        distance = args['--distance']        
        
        model_save_to = os.path.join(output_folder, model_name_save_to)
        
        if dataset == 'sts':
            sent_bert.train_sentence_bert_sts(
                device = device,
                model_save_to = model_save_to,
                normalize_embeddings = normalize_embeddings, 
                center_embeddings = center_embeddings,
                z_score_norm = z_score_norm,
                distance = distance,
                pretrained_model_name = pretrained_model_name,
                train_batch_size = train_batch_size,
                num_epochs = num_epochs)
            
        elif dataset == 'newsgroups':
            print(args)
            triplet_margin = float(args['--triplet-margin'])
            train_dataset_type = args['--train-dataset-type']
            output_embedding_size = int(args['--embedding-size'])
            
            max_seq_length = int(args['--max-seq-length']) 
            
            sent_bert.train_sentence_bert_newsgroups(
                device = device,
                model_save_to = model_save_to,
                normalize_embeddings = normalize_embeddings,
                center_embeddings = center_embeddings,
                z_score_norm = z_score_norm,
                triplet_margin = triplet_margin,
                train_dataset_type = train_dataset_type,
                distance = distance,
                pretrained_model_name = pretrained_model_name,
                max_seq_length = max_seq_length, 
                output_embedding_size = output_embedding_size,
                train_batch_size = train_batch_size,
                num_epochs = num_epochs)
        
        else:
            raise NotImplementedError(f'Dataset not implemented for training: {dataset}')
    

def get_embeddings(args: Dict) -> None:
    
    model_names_path = args['--model-names-path'] if args['--model-names-path'] else ''
    model_name = args['--model-name'] if args['--model-name'] else ''
    output_folder = args['--output-folder'] if args['--output-folder'] else ''   
    emb_save_to = args['--embeddings-save-to']
    
    if not model_name in sent_bert.get_pretrained_sentence_bert_model_names():
        model_load_path = os.path.join(output_folder, naming.get_model_folder(), model_name)
    else:
        model_load_path = model_name
    
    dataset, subset = args['--dataset'].split('.')
    
    if args['--sentences']:
        sentences = loadd.load_sentences_csv_file(args['--sentences'])    
    else:
        sentences = loadd.load_sentences_from_known_dataset(
            dataset_folder = 'datasets',
            dataset = dataset, 
            subset = subset)  
    
    device = "cuda" if args['--cuda'] else "cpu"
    
    if model_names_path == 'models':
        sent_bert.get_and_save_sentence_bert_embeddings_from_model_folder(
            output_folder = output_folder,
            model_prefix = model_name,
            dataset = dataset,
            subset = subset,
            sentences = sentences,
            device = device)  
    else:
        sent_bert.get_and_save_sentence_bert_embeddings(
            sentences = sentences, 
            model_name = model_load_path,
            save_to = emb_save_to,
            device = device)


def get_performance(args: Dict):
    
    seed = int(args['--seed'])
    folds = int(args['--num-fold'])
    n_jobs = int(args['--n-jobs'])
    center_norm_after_train = True if args['--center-norm-after-train'] else False
    
    if ',' in args['--force-dist']:
        force_dists = [ForceDistribution(f) 
                       for f in args['--force-dist'].split(',')]
    else:
        force_dists = [ForceDistribution(args['--force-dist'])]
    
    if ',' in args['--skhubness-reduction']:
        skhubness_reductions = [SkhubnessReduction(f) 
                       for f in args['--skhubness-reduction'].split(',')]
    else:
        skhubness_reductions = [SkhubnessReduction(args['--skhubness-reduction'])]
       
    
    result_type = args['--result-type']
    
    output_folder = args['--output-folder'] if args['--output-folder'] else ''
    model_names_path = args['--model-names-path'] if args['--model-names-path'] else ''
    model_name_prefix = args['--model-name'] if args['--model-name'] else ''
    
    dataset, subset = args['--dataset'].split('.')
    
    device = "cuda" if args['--cuda'] else "cpu"
    
    if result_type == 'knn':
        for force_dist in force_dists:
            logging.info(f'Will use force_dist: {force_dist.name}')
            for skhubness_reduction in skhubness_reductions:
                logging.info(f'Will use skhubness_reduction: {skhubness_reduction.name}')
                if model_names_path == 'embeddings':
                    savr.get_d_fold_performance_of_models_from_embedding_files(
                        output_folder = output_folder, 
                        dataset = dataset,
                        device = device,
                        random_state = seed,
                        center_norm_after_train = center_norm_after_train,
                        skhubness_reduction = skhubness_reduction,
                        force_dist = force_dist,
                        d = folds,
                        n_jobs = n_jobs)
                elif model_names_path == 'models':
                    savr.get_d_fold_performance_of_models_from_model_folder(
                        output_folder = output_folder,
                        model_prefix = model_name_prefix,
                        dataset = dataset,
                        device = device,
                        random_state = seed,
                        center_norm_after_train = center_norm_after_train,
                        skhubness_reduction = skhubness_reduction,
                        force_dist = force_dist,
                        d = folds,
                        n_jobs = n_jobs)
                elif model_names_path != '':
                    savr.get_d_fold_performance_of_models_from_file(
                        output_folder = output_folder, 
                        model_names_path = model_names_path, 
                        dataset = dataset,
                        device = device,
                        random_state = seed,
                        center_norm_after_train = center_norm_after_train,
                        skhubness_reduction = skhubness_reduction,
                        force_dist = force_dist,
                        d = folds, 
                        get_class_scores_fold = False,
                        get_class_scores_test = True,
                        n_jobs = n_jobs)
                else:
                    raise ValueError('model_names_path was empty')
    
    if result_type == 'hubness':
        if model_names_path == 'embeddings':
            savr.get_hubness_from_embedding_files(
                    output_folder = output_folder,
                    dataset = dataset,
                    random_state = seed,
                    skhubness_reductions = skhubness_reductions,
                    force_dists = force_dists)
        else:
            raise ValueError("hubness results only works for model_names_path == 'embeddings'")


def main():
    
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print info to stdout
    
    args = docopt(__doc__)
        
    # seed the random number generators
    seed = int(args['--seed'])
    use_cuda = True if args['--cuda'] else False     
    random_util.set_seed(seed, use_cuda)

    if args['train']:
        train(args)
    elif args['embeddings']:
        get_embeddings(args)
    elif args['performance']:
        get_performance(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()


