# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:54:41 2022

@author: bemigini


Loading various datasets for usage later.


"""

import csv
import gzip
import numpy as np
from numpy.typing import NDArray
import os
import pandas as pd


from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.readers import InputExample
from sklearn.datasets import fetch_20newsgroups

from src.datasets.SampleTripletDataset import SampleTripletDataset
from src.datasets.CompleteTripletDataset import CompleteTripletDataset

from typing import Iterable, NamedTuple, Tuple


class RawSentence(NamedTuple):
    text: str
    sentence_id: int
    document_id: int


def get_doc_sent_id(doc_id: int, sentence_id: int) -> str:
    return 'd{0}.s{1}'.format(doc_id, sentence_id)


def get_instance_id(doc_id: int, sentence_id: int, instance_index: int) -> str:
    return '{0}.t{1}'.format(
        get_doc_sent_id(doc_id, sentence_id), 
        instance_index)

def get_doc_sent_id_from_instance_id(instance_id: str) -> str:
    return instance_id.split('.t')[0]

def get_token_index(instance_id: str) -> int:
    return int(instance_id.split('.t')[1])


def read_separated_file_pd(
        sent_dist_folder, path_from_sent_dist, encoding, delimiter = '\t', 
        header = 0, names = None, quoting = csv.QUOTE_NONE):
    full_path = os.path.join(sent_dist_folder, path_from_sent_dist)
    
    # Quote marks should just be treated as part of the string.
    dialect = csv.excel()
    dialect.quoting = quoting
    dialect.delimiter = delimiter    
    df = pd.read_csv(full_path, dialect = dialect, encoding = encoding, 
                     header = header, names = names, on_bad_lines = 'warn')
            
    return df


def read_separated_file(sent_dist_folder, path_from_sent_dist, encoding, delimiter = '\t'):
    full_path = os.path.join(sent_dist_folder, path_from_sent_dist)
    
    with open(full_path, 'r', encoding = encoding) as f:
        text_lines = f.readlines()
    
    records = [line.replace('\n', '').split(delimiter) for line in text_lines]
    
    return records


# https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark
def read_sts_benchmark_file(sent_dist_folder, file_type, use_pd = True):
    path = 'datasets/stsbenchmark/sts-{0}.csv'.format(file_type)
    sts_columns = ['genre', 
                   'filename', 
                   'year', 
                   'id', 
                   'score', 
                   'sentence1', 
                   'sentence2', 
                   'url1', 
                   'url2']
    
    if use_pd:
        data = read_separated_file_pd(
                sent_dist_folder, 
                path, 
                encoding = 'utf-8', 
                delimiter = '\t', 
                header = None, 
                names = sts_columns)
    else:
        records = read_separated_file(
            sent_dist_folder, 
            path, 
            encoding = 'utf-8', 
            delimiter = '\t')
        
        data = pd.DataFrame.from_records(
            records, 
            columns = sts_columns)
        
    return data


# https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark
def load_sts_benchmark(sent_dist_folder, use_pd = True):
    types = ['dev', 'train', 'test']
    
    dev = read_sts_benchmark_file(sent_dist_folder, types[0], use_pd)
    train = read_sts_benchmark_file(sent_dist_folder, types[1], use_pd)
    test = read_sts_benchmark_file(sent_dist_folder, types[2], use_pd)
    
    return dev, train, test


def load_sts_benchmark_combined(sentence_dist_folder: str) -> pd.DataFrame:
    dev, train, test = load_sts_benchmark(sentence_dist_folder)
    dev_train_test_col = 'dev_train_or_test'
    dev.insert(len(dev.columns), dev_train_test_col, 'dev')
    train.insert(len(train.columns), dev_train_test_col, 'train')
    test.insert(len(test.columns), dev_train_test_col, 'test')
    sts_df = pd.concat([dev, train, test], ignore_index = True)    
    
    return sts_df


def save_sts_benchmark_combined_with_sent_ids(
        sentence_dist_folder: str,
        save_to_folder: str = 'datasets/stsbenchmark') -> None:
    sts_df = load_sts_benchmark_combined(sentence_dist_folder)
    
    sentence1_id = [int(str(idx) + str(1)) for idx in sts_df.index]
    sentence2_id = [int(str(idx) + str(2)) for idx in sts_df.index]
    
    sts_df.insert(1, 'sentence1_id', sentence1_id)
    sts_df.insert(2, 'sentence2_id', sentence2_id)
    
    save_file_path = os.path.join(sentence_dist_folder, save_to_folder, 'sts_combined_with_sent_ids.csv')
    sts_df.to_csv(save_file_path, index = False)
    
    
def load_sts_benchmark_combined_with_sent_ids(
        sentence_dist_folder: str, 
        data_folder: str = 'datasets/stsbenchmark') -> pd.DataFrame:
        
    file_path = os.path.join(sentence_dist_folder, data_folder, 'sts_combined_with_sent_ids.csv')
    sts_df = pd.read_csv(file_path)
    
    return sts_df
    

def read_gzip_file(path):
    with gzip.open(path, 'rb') as file:
        read_file_str = file.read()
    
    return read_file_str


def make_sts_sentence_id_df(
        sentence_dist_folder: str) -> pd.DataFrame:
    sts_df = load_sts_benchmark_combined_with_sent_ids(sentence_dist_folder)
    
    sentence1_id = sts_df['sentence1_id'].values
    sentence2_id = sts_df['sentence2_id'].values
    sentence1 = sts_df['sentence1'].values
    sentence2 = sts_df['sentence2'].values
    
    sentence_dict = {
        'id': np.append(sentence1_id, sentence2_id), 
        'sentence': np.append(sentence1, sentence2)}
    
    sentence_df = pd.DataFrame(data = sentence_dict)
    
    return sentence_df
    

def save_sts_sentence_id_csv(
        sentence_dist_folder: str, 
        save_to_folder: str = 'datasets/stsbenchmark') -> None:
    
    sentence_df = make_sts_sentence_id_df(sentence_dist_folder)
    save_file_path = os.path.join(save_to_folder, 'sts_sentence_ids.csv')
    sentence_df.to_csv(save_file_path, index = False)


def load_sts_sentence_id_csv(
        dataset_folder: str) -> pd.DataFrame:    
    
    file_path = os.path.join(dataset_folder, 'stsbenchmark/sts_sentence_ids.csv')
    
    sentence_df = pd.read_csv(file_path, index_col = False)
    
    return sentence_df


def get_20_newsgoups_data(subset: str):
    newsgroups_no_meta = fetch_20newsgroups(
        subset = subset, 
        remove=('headers', 'footers', 'quotes'))
    
    return newsgroups_no_meta
    

def get_SentenceLabelDataSet_from_newsgroups(
        subset: str = 'train',
        samples_per_label: int = 2,
        with_replacement: bool = False):
    
    newsgroups_no_meta = get_20_newsgoups_data(subset)
    
    input_examples = [InputExample(
        texts=[newsgroups_no_meta.data[i]], 
        label=newsgroups_no_meta.target[i])
        for i in range(len(newsgroups_no_meta.data))]
    
    dataset = SentenceLabelDataset(input_examples, 
                                   samples_per_label = samples_per_label,
                                   with_replacement = with_replacement)
    
    return dataset


def get_train_dev_TripletDataset_from_newsgroups(
        triplets_per_anchor: int = 2,
        with_replacement: bool = False,
        train_dataset_type: str = 'many'):
    
    newsgroups_no_meta = get_20_newsgoups_data('train')
    
    dev_indexes = []
    for i in range(len(newsgroups_no_meta.target_names)):
        target_indexes = np.where(newsgroups_no_meta.target == i)[0]
        chosen_for_dev = np.random.choice(target_indexes, size = 5, replace = False)
        dev_indexes.extend(chosen_for_dev)
    
    dev_examples = [InputExample(
        texts=[newsgroups_no_meta.data[i]], 
        label=newsgroups_no_meta.target[i])
        for i in dev_indexes]
    
    train_examples = [InputExample(
        texts=[newsgroups_no_meta.data[i]], 
        label=newsgroups_no_meta.target[i])
        for i in range(len(newsgroups_no_meta.data)) 
        if i not in dev_indexes]
    
    dev_dataset = CompleteTripletDataset(dev_examples, 
                                         make_all_combinations = True,
                                         shuffle = False,
                                         chunks = 1)
    
    if train_dataset_type == 'sample':
        print('sample')
        train_dataset = SampleTripletDataset(train_examples, 
                                   triplets_per_anchor = triplets_per_anchor,
                                   with_replacement = with_replacement)
    elif train_dataset_type == 'many':
        print('many combinations')
        train_dataset = CompleteTripletDataset(train_examples, 
                                             make_all_combinations = False,
                                             shuffle = True,
                                             chunks = 1)  
    else:
        print('all combinations')
        train_dataset = CompleteTripletDataset(train_examples, 
                                             make_all_combinations = True,
                                             shuffle = True,
                                             chunks = 10)  
    
    print(len(train_dataset))
    
    return train_dataset, dev_dataset


def load_train_test_class_index_to_name(folder: str):
    train_file = 'train.csv.gz'
    test_file = 'test.csv.gz'
    class_index_to_name_file = 'class_index_to_name.csv'
    
    train_df = pd.read_csv(os.path.join(folder, train_file))
    test_df = pd.read_csv(os.path.join(folder, test_file))  
    train_df = train_df.fillna('')
    test_df = test_df.fillna('')
    
    class_index_to_name = np.loadtxt(
        os.path.join(folder, class_index_to_name_file),
        dtype = str,
        delimiter = ',',
        skiprows = 1)
    
    return train_df, test_df, class_index_to_name


def load_ag_news(dataset_folder: str) -> Tuple[NDArray, NDArray, NDArray]:
    folder = os.path.join(dataset_folder, 'AG_news_classification')
    
    train_df, test_df, class_index_to_name = load_train_test_class_index_to_name(folder)
    
    train_df['text'] = (train_df['Title'] + ' ' + train_df['Description']).str.replace('\\', ' ', regex = False)
    test_df['text'] = (test_df['Title'] + ' ' + test_df['Description']).str.replace('\\', ' ', regex = False)
    
    train = train_df.to_numpy()
    test = test_df.to_numpy()
    
    return train, test, class_index_to_name


def load_yahoo_answers(dataset_folder: str) -> Tuple[NDArray, NDArray, NDArray]:
    folder = os.path.join(dataset_folder, 'Yahoo_answers_10_categories')
    
    train_df, test_df, class_index_to_name = load_train_test_class_index_to_name(folder)

    train_df['text'] = (train_df['question_title'] + ' ' + train_df['question_content'] + ' ' + train_df['best_answer']).str.replace('\\', ' ', regex = False)
    test_df['text'] = (test_df['question_title'] + ' ' + test_df['question_content'] + ' ' + test_df['best_answer']).str.replace('\\', ' ', regex = False)
    
    train = train_df.to_numpy()
    test = test_df.to_numpy()
    
    return train, test, class_index_to_name


def load_yahoo_answers_small(
        dataset_folder: str,
        frac: float,
        random_state: int) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    folder = os.path.join(dataset_folder, 'Yahoo_answers_10_categories')
    
    train_df, test_df, class_index_to_name = load_train_test_class_index_to_name(folder)
    
    train_small_df = pd.DataFrame(columns = train_df.columns)
    test_small_df = pd.DataFrame(columns = test_df.columns)
    train_indexes = np.array([], dtype = int)
    test_indexes = np.array([], dtype = int)
    
    for class_index in class_index_to_name[:, 0]:
        class_index_int = int(class_index)
        train_w_class = train_df[train_df['class_index'] == class_index_int]
        test_w_class = test_df[test_df['class_index'] == class_index_int]
        
        train_w_class_sample = train_w_class.sample(frac = frac, random_state = random_state)
        test_w_class_sample = test_w_class.sample(frac = frac, random_state = random_state)
        
        train_small_df = pd.concat([train_small_df, train_w_class_sample], ignore_index=True)
        test_small_df = pd.concat([test_small_df, test_w_class_sample], ignore_index=True)
        
        train_indexes = np.concatenate(
            (train_indexes, train_w_class_sample.index.to_numpy()))
        test_indexes = np.concatenate(
            (test_indexes, test_w_class_sample.index.to_numpy()))        
       
    
    train_small_df['text'] = (train_small_df['question_title'] + ' ' + train_small_df['question_content'] + ' ' + train_small_df['best_answer']).str.replace('\\n', ' ', regex = False)
    test_small_df['text'] = (test_small_df['question_title'] + ' ' + test_small_df['question_content'] + ' ' + test_small_df['best_answer']).str.replace('\\n', ' ', regex = False)
    
    train = train_small_df.to_numpy()
    test = test_small_df.to_numpy()
    
    return train, test, class_index_to_name, train_indexes, test_indexes


def get_sentences_from_dataset(
        dataset_folder: str, 
        dataset: str,
        subset: str) -> NDArray[str]:
    
    if dataset == 'ag_news':
        train, test, class_index_to_name = load_ag_news(dataset_folder)
    elif dataset == 'yahoo_answers':
        train, test, class_index_to_name = load_yahoo_answers(dataset_folder)
    elif dataset == 'yahoo_answers_small':
        train, test, class_index_to_name, _, _ = load_yahoo_answers_small(
            dataset_folder, frac = 0.1, random_state = 0)
    else:
        raise ValueError(f'dataset not recognized: {dataset}')
    
    if subset == 'train':
        return train[:, -1]
    elif subset == 'test':
        return test[:, -1]
    
    return np.concatenate((train[:, -1],test[:, -1]))


def load_targets_from_dataset(
        dataset_folder: str, 
        dataset: str,
        subset: str) -> NDArray[str]:
    
    if dataset == 'newsgroups':
        if subset == 'train' or subset == 'test':
            newsgroups_no_meta = fetch_20newsgroups(
                subset = subset, 
                remove=('headers', 'footers', 'quotes'))        
    
            targets = newsgroups_no_meta.target
        else:
            newsgroups_train_no_meta = fetch_20newsgroups(
                subset = 'train', 
                remove=('headers', 'footers', 'quotes'))        
    
            train_targets = newsgroups_train_no_meta.target
            
            newsgroups_test_no_meta = fetch_20newsgroups(
                subset = 'test', 
                remove=('headers', 'footers', 'quotes'))        
    
            test_targets = newsgroups_test_no_meta.target
            
            targets = np.concatenate((train_targets,test_targets))
            
        return targets
    
    elif dataset == 'ag_news':
        train, test, class_index_to_name = load_ag_news(dataset_folder)        
    elif dataset == 'yahoo_answers':
        train, test, class_index_to_name = load_yahoo_answers(dataset_folder)
    elif dataset == 'yahoo_answers_small':
        train, test, class_index_to_name, _, _ = load_yahoo_answers_small(
            dataset_folder, frac = 0.1, random_state = 0)
    else:
        raise ValueError(f'dataset not recognized: {dataset}')
        
    if subset == 'train':
        return train[:, 0]
    elif subset == 'test':
        return test[:, 0]
    
    return np.concatenate((train[:, 0],test[:, 0]))


def load_sentences_csv_file(file_path: str) -> Iterable[str]:
    sentences = np.loadtxt(file_path, dtype = str, delimiter = ',')
    
    return sentences
    

def load_sentences_from_known_dataset(
        dataset_folder: str = 'datasets',
        dataset: str = 'sts', 
        subset: str = 'all') -> Iterable[str]:
    
    if dataset == 'sts':
        if subset == 'all':
            sentence_df = load_sts_sentence_id_csv(dataset_folder)
            sentences = sentence_df['sentence'].values
        else:
            raise NotImplementedError('Getting only train, dev or test sts sentences not implemented')
    elif dataset == 'newsgroups':
        newsgroups_no_meta = get_20_newsgoups_data(subset)
        sentences = newsgroups_no_meta.data
    elif dataset == 'ag_news' or dataset == 'yahoo_answers' or dataset == 'yahoo_answers_small':
        sentences = get_sentences_from_dataset(dataset_folder, dataset, subset)
    
    return sentences












