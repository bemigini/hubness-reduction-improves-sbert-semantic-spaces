# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 15:20:14 2022

@author: bmgi


Default folders and names for models, embeddings and results


"""


import os
import re

from typing import List, Tuple



def get_model_folder():
    return 'models'

def get_embedding_folder():
    return 'embeddings'

def get_results_folder():
    return 'results'


def get_sts_bert_model_save_name(
        base_model: str,
        distance: str,
        max_dist_relation: str,
        z_score_norm: bool,
        normalize_embeddings: bool,
        center_embeddings: bool,
        seed: int):
    return f'sts_bert_{base_model}_{distance}_{max_dist_relation}_z_{z_score_norm}_n_{normalize_embeddings}_c_{center_embeddings}_seed{seed}'


def get_model_names_from_model_folder(
        output_folder: str,
        prefix: str = '') -> List[str]:
    model_dir_path = os.path.join(output_folder, get_model_folder())
    
    model_names = [folder 
                    for folder in os.listdir(model_dir_path)
                    if re.match(f'{prefix}.*', folder)]
    
    return model_names


def get_model_names_from_embeddings(
        output_folder: str, 
        dataset: str,
        embedding_folder: str = '') -> List[str]:
    
    if embedding_folder == '':
        embedding_dir_path = os.path.join(output_folder, get_embedding_folder())
    else:
        embedding_dir_path = os.path.join(output_folder, embedding_folder)
    
    embedding_files = [file 
                       for file in os.listdir(embedding_dir_path)
                       if re.match(f'.*_{dataset}_.*\.h5.*', file)]
    model_names = list(set([file.split(f'_{dataset}_')[0] 
                            for file in embedding_files]))
    
    return model_names


def get_model_embedding_name(
        model_name: str, 
        dataset:str, 
        subset: str) -> str:
    model_embedding_name = f'{model_name}_{dataset}_{subset}.h5'
    
    return model_embedding_name


def check_folders(output_folder: str) -> Tuple[str, str]:
    embedding_folder_path = os.path.join(output_folder, get_embedding_folder())
    os.makedirs(embedding_folder_path, exist_ok = True)
        
    results_folder_path = os.path.join(output_folder, get_results_folder())
    os.makedirs(results_folder_path, exist_ok = True)
    
    return embedding_folder_path, results_folder_path


def get_model_names_from_knn_results_and_results_path(
        results_dir_path: str,
        dataset: str,
        prefix: str = ''):
    
    files_in_results = [file 
                        for file in os.listdir(results_dir_path)
                        if re.match(f'{prefix}.*_{dataset}_knn_results_.*\.json', file)]
    
    model_names = list(set([re.split(f'_{dataset}_', file)[0] 
                                 for file in files_in_results]))
    
    return model_names


def get_model_names_from_knn_results(
        output_folder: str,
        dataset: str,
        prefix: str = '',
        results_folder: str = '') -> List[str]:
    if results_folder == '':
        results_dir_path = os.path.join(output_folder, get_results_folder())
    else:
        results_dir_path = os.path.join(output_folder, results_folder)
        
        
    model_names = get_model_names_from_knn_results_and_results_path(
        results_dir_path,
        dataset,
        prefix)
    
    return model_names


def get_model_name_roots_from_knn_results(
        output_folder: str,
        dataset: str,
        prefix: str = '',
        results_folder: str = '') -> List[str]:
    model_names = get_model_names_from_knn_results(
        output_folder, dataset, prefix, results_folder)
    
    model_name_roots = list(set([re.split('_seed\d+', name)[0] 
                                 for name in model_names]))
    
    return model_name_roots


def get_base_model_from_model_name(model_name: str):
    return model_name.replace('sts_bert_', '').split('_')[0]


def get_distance_from_model_name(model_name: str):
    if '_cos_dist_' in model_name:
        return 'cos_dist'
    if '_euclidean_' in model_name:
        return 'euclidean'
    if '_cos_' in model_name:
        return 'cos'
    
    return ''


def get_train_norm_from_model_name(model_name: str):
    train_norms = []
    
    if '_c_True' in model_name:
        train_norms.append('c')
    if '_n_True' in model_name:
        train_norms.append('n')
    if '_z_True' in model_name:
        train_norms.append('z')
    
    if train_norms:
        return ', '.join(train_norms)
    
    return 'none'


def get_seed_from_model_name(model_name: str):
    return model_name.split('_seed')[1]




