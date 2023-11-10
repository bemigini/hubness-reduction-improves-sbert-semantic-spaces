# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 09:02:16 2022

@author: bmgi


k-nearest neighbours performance of various models 


"""


from dataclasses import dataclass

import logging
import numpy as np
from numpy.typing import NDArray

import os

from scipy.sparse import csr_matrix

# from skhubness.neighbors import NMSlibTransformer
# Made custom code to handle when not all neighbours are found
# from src.skhubness_custom.nmslib_wrap import NMSlibTransformer
from src.skhubness_custom import nmslib_wrap

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from src import sentence_bert as sent_bert
from src.datasets import loading_datasets as loadd
from src.util import naming, summary_stats
from src.hubness import analysis
import src.hubness.reduction as reduction
from src.hubness.reduction import ForceDistribution, SkhubnessReduction

from tqdm import tqdm
from typing import Dict, Iterable, List, Tuple






@dataclass
class KnnScores():
    train_score: float 
    test_score: float
    correct_test_idx: NDArray
    

@dataclass
class FeaturesTargetsEmbeddings():
    features: NDArray
    targets: NDArray
    embeddings: NDArray
    
    
@dataclass
class FewShotFilters():
    target_filter: NDArray 
    no_target_filter: NDArray
    few_target_indexes: NDArray
    


def get_knn_graph(
        neighbour_num: int,
        train_features: NDArray,        
        test_features: NDArray,
        output_folder: str,
        knn_metric: str = 'euclidean',
        knn_mode: str = 'distance',
        n_jobs: int = 5) -> Tuple[csr_matrix, csr_matrix]:
    logging.info('In get_knn_graph')
    logging.info(f'Train shape: {train_features.shape}, test shape: {test_features.shape}')
    
    knn_trans = analysis.get_knn_transformation(
            neighbour_num,
            train_features,
            output_folder,
            knn_metric,
            knn_mode,
            n_jobs)
    
    graph_train: csr_matrix = knn_trans.transform(train_features)
    graph_test: csr_matrix = knn_trans.transform(test_features)
    
    return graph_train, graph_test


def fit_nearest_neighbour_from_graph_and_get_score(
        neighbour_num: int, 
        graph_train: csr_matrix, 
        train_targets: NDArray, 
        graph_test: csr_matrix, 
        test_targets: NDArray,
        knn_mode: str = 'distance') -> Tuple[KNeighborsClassifier, float, float, NDArray]:
    
    knn_clf = KNeighborsClassifier(
        n_neighbors=neighbour_num,
        weights = knn_mode,
        metric='precomputed')
    knn_clf.fit(graph_train, train_targets)
    
    predicted = knn_clf.predict(graph_test)
    correct_test_idx = np.nonzero(predicted == test_targets)
    
    train_score = knn_clf.score(graph_train, train_targets)
    test_score = knn_clf.score(graph_test, test_targets)
    
    return knn_clf, train_score, test_score, correct_test_idx


def check_graphs_for_nan_and_inf(train_graph: csr_matrix, test_graph: csr_matrix):
    if np.isnan(train_graph.data).any() or np.isinf(train_graph.data).any():
        logging.warning('Nan or inf found in training graph. Nan values will be replaced by zero.')
        train_graph.data = np.nan_to_num(train_graph.data)
    if np.isnan(test_graph.data).any() or np.isinf(test_graph.data).any():
        logging.warning('Nan or inf found in test graph. Nan values will be replaced by zero.')
        test_graph.data = np.nan_to_num(test_graph.data)
    
    return train_graph, test_graph


def get_nearest_neighbour_results_graph(
        train_data: NDArray, 
        train_targets: NDArray, 
        test_data: NDArray, 
        test_targets: NDArray,
        output_folder: str,
        num_neighbours: Iterable[int],
        skhubness_reduction: SkhubnessReduction = SkhubnessReduction.NONE,
        get_class_scores: bool = False,
        n_jobs: int = 5,
        get_correct_test_idx: bool = False) -> Dict[str, KnnScores]:
    
    unique_targets = np.unique(train_targets)
    
    max_num_neighbours = max(num_neighbours)
    
    if skhubness_reduction == SkhubnessReduction.MUTUAL_PROXIMITY:
        # To get stable results with Mutual proximity we set a larger number 
        # of neighbours, but we won't go above number of training samples. 
        max_num_neighbours = min(max_num_neighbours + 100, train_data.shape[0])
        logging.info(f'max_num_neighbours: {max_num_neighbours}')
    
    knn_mode = 'distance'
    graph_train, graph_test = get_knn_graph(
        neighbour_num = max_num_neighbours, 
        train_features = train_data,        
        test_features = test_data,
        output_folder = output_folder,
        knn_metric = 'euclidean',
        knn_mode = knn_mode,
        n_jobs = n_jobs)
    
    if skhubness_reduction == SkhubnessReduction.MUTUAL_PROXIMITY:
        graph_train, graph_test = reduction.get_mutual_proximity_graphs(
            graph_train, graph_test)
        
    if skhubness_reduction == SkhubnessReduction.LOCAL_SCALING:
        graph_train, graph_test = reduction.get_local_scaling_graphs(
            graph_train, graph_test)
    
    graph_train, graph_test = check_graphs_for_nan_and_inf(graph_train, graph_test)
    
    results = {}
    for neighbour_num in tqdm(num_neighbours):
        
        n_classifier, train_score, test_score, correct_test_idx = fit_nearest_neighbour_from_graph_and_get_score(
            neighbour_num = neighbour_num, 
            graph_train = graph_train, 
            train_targets = train_targets, 
            graph_test = graph_test, 
            test_targets = test_targets,
            knn_mode = knn_mode)
        
        if not get_correct_test_idx:
            correct_test_idx = np.array([])
        
        knn_name = f'{knn_mode}_{neighbour_num}'            
        results[knn_name] = KnnScores(train_score, test_score, correct_test_idx)
        
        if get_class_scores:
            for target in unique_targets:
                train_target_filter =  train_targets == target
                test_target_filter = test_targets == target
                
                target_train_score = n_classifier.score(
                    graph_train[train_target_filter], 
                    train_targets[train_target_filter])
                target_test_score = n_classifier.score(
                    graph_test[test_target_filter], 
                    test_targets[test_target_filter])
                
                target_knn_name = f'{knn_mode}_{neighbour_num}_t_{target}'
                results[target_knn_name] = KnnScores(target_train_score, target_test_score, np.array([]))
        
    
    return results   


def get_num_neighbours_from_knn_result_key(key: str) -> int:
    return int(key.split('_')[-1])
    

def get_highest_train_and_test_result_keys(
        results: Dict[str, KnnScores]) -> Tuple[str, str]:
    max_train_key = max(results, key = lambda k: results.get(k).train_score)
    max_test_key = max(results, key = lambda k: results.get(k).test_score)
    
    return max_train_key, max_test_key


def get_train_test_summaries(
        train_test_scores = List[KnnScores],
        return_scores: bool = False) -> Tuple[summary_stats.SummaryStats, summary_stats.SummaryStats]:
    train_scores = [score.train_score for score in train_test_scores]
    test_scores = [score.test_score for score in train_test_scores]
    
    train_summary = summary_stats.SummaryStats(train_scores)
    test_summary = summary_stats.SummaryStats(test_scores)
    
    if return_scores:
        return train_summary, test_summary, train_scores, test_scores
    
    return train_summary, test_summary


def get_features_and_targets(
        dataset: str, 
        subsets: List[str],
        data_dict: Dict[str, FeaturesTargetsEmbeddings]) -> Dict[str, FeaturesTargetsEmbeddings]:
    
    if dataset == 'newsgroups':
        for subset in subsets:
            data = loadd.get_20_newsgoups_data(subset)
            features = data.data 
            targets = data.target
            data_dict[subset] = FeaturesTargetsEmbeddings(features, targets, None)
    else:
        if dataset == 'ag_news':
            train, test, class_index_to_name = loadd.load_ag_news(dataset_folder = 'datasets')            
        elif dataset == 'yahoo_answers':
            train, test, class_index_to_name = loadd.load_yahoo_answers(dataset_folder = 'datasets')
        elif dataset == 'yahoo_answers_small':
            train, test, class_index_to_name, _, _ = loadd.load_yahoo_answers_small(
                dataset_folder = 'datasets', frac = 0.1, random_state = 0)
        else:
             raise NotImplementedError(f'Dataset not implemented: {dataset}')
        
        train_features = train[:, -1]
        test_features = test[:, -1]
        train_targets = train[:, 0].astype(int)
        test_targets = test[:, 0].astype(int)
        data_dict['train'] = FeaturesTargetsEmbeddings(train_features, train_targets, None)
        data_dict['test'] = FeaturesTargetsEmbeddings(test_features, test_targets, None)       
    
    return data_dict


def get_all_embeddings(
        output_folder: str, 
        model_name: str, 
        dataset: str,
        subsets: List[str],
        device: str,
        embedding_folder_path: str,
        data_dict: Dict[str, FeaturesTargetsEmbeddings]) -> Dict[str, FeaturesTargetsEmbeddings]:    
        
    for subset in subsets:
        # Read embeddings from hdf5 file
        emb_name = naming.get_model_embedding_name(model_name, dataset, subset)
        
        model_embedding_path = os.path.join(embedding_folder_path, emb_name)
                
        if not os.path.exists(model_embedding_path):
            model_load_path = os.path.join(output_folder, naming.model_folder, model_name)                
            
            features = data_dict[subset].features
            embeddings = sent_bert.get_and_save_sentence_bert_embeddings(
                features,
                model_load_path,
                save_to = model_embedding_path,
                device = device)
            
        else:
            embeddings = sent_bert.load_sentence_bert_embeddings(model_embedding_path)
        
        data_dict[subset].embeddings = embeddings    
    
    return data_dict


def get_d_fold_results(
        all_train_embeddings: NDArray,
        all_train_targets: NDArray,
        test_embeddings: NDArray,
        test_targets: NDArray,
        output_folder: str,
        num_neighbours: Iterable[int],
        random_state: int,
        center_norm_after_train: bool,
        skhubness_reduction: SkhubnessReduction = SkhubnessReduction.NONE,
        force_dist: ForceDistribution = ForceDistribution.NONE,
        d: int = 5, 
        get_class_scores_fold: bool = False,
        get_class_scores_test: bool = True,
        n_jobs: int = 5) -> Dict[str, object]:
    
    knn_fold_results = {}     
        
    if force_dist == ForceDistribution.NORMAL_ALL or force_dist == ForceDistribution.UNIFORM_ALL:
        all_train_embeddings_not_dist = np.copy(all_train_embeddings)
        logging.info(f'Force dist: {force_dist.name}, {force_dist.value}')
        logging.info(f'All train shape before transform: {all_train_embeddings.shape}')
        
        if force_dist == ForceDistribution.NORMAL_ALL:
            distribution_type = reduction.ProbabilityDistribution.NORMAL
        if force_dist == ForceDistribution.UNIFORM_ALL:
            distribution_type = reduction.ProbabilityDistribution.UNIFORM
        
        all_train_embeddings = reduction.transform_dimensions(
            all_train_embeddings,
            random_state,
            distribution_type = distribution_type,
            normalize_rows = True)
        logging.info(f'All train shape after transform: {all_train_embeddings.shape}')
        
    
    skf = StratifiedKFold(n_splits = d, shuffle = True, random_state = random_state)
    d_train_test_splits = skf.split(all_train_embeddings, all_train_targets)
    
    for i, (train_idx, test_idx) in enumerate(d_train_test_splits):        
        d_train_features = all_train_embeddings[train_idx]
        d_test_features = all_train_embeddings[test_idx]
        d_train_targets = all_train_targets[train_idx]
        d_test_targets = all_train_targets[test_idx]
        
        if center_norm_after_train:
            d_train_features, d_test_features = reduction.center_dimensions_normalize_embeddings(d_train_features, d_test_features)
        elif force_dist == ForceDistribution.NORMAL_SPLITS:
            d_train_features = reduction.transform_dimensions(
                d_train_features, 
                random_state,
                distribution_type = reduction.ProbabilityDistribution.NORMAL,
                normalize_rows = True)
            d_test_features = reduction.transform_dimensions(
                d_test_features, 
                random_state,
                distribution_type = reduction.ProbabilityDistribution.NORMAL,
                normalize_rows = True)
        elif force_dist == ForceDistribution.UNIFORM_SPLITS:
            d_train_features = reduction.transform_dimensions(
                d_train_features, 
                random_state,
                distribution_type = reduction.ProbabilityDistribution.UNIFORM,
                normalize_rows = True)
            d_test_features = reduction.transform_dimensions(
                d_test_features, 
                random_state,
                distribution_type = reduction.ProbabilityDistribution.UNIFORM,
                normalize_rows = True)
            
        
        knn_results = get_nearest_neighbour_results_graph(
            d_train_features,
            d_train_targets,
            d_test_features,
            d_test_targets,
            output_folder,
            num_neighbours,
            skhubness_reduction = skhubness_reduction,
            get_class_scores = get_class_scores_fold,
            n_jobs = n_jobs,
            get_correct_test_idx = False)
        
        base_knn_results = {r: knn_results[r] for r in knn_results if '_t_' not in r}        
        max_train_key, max_test_key = get_highest_train_and_test_result_keys(base_knn_results)                
        
        knn_fold_results[f'max_train_key_fold_{i}'] = max_train_key
        knn_fold_results[f'max_test_key_fold_{i}'] = max_test_key
        knn_fold_results[f'knn_results_fold_{i}'] = knn_results
        
    
    # Get performance of number of neighbours over folds
    neighbour_perf = {}
    for neighbour_num in num_neighbours:
        neighbour_perf[neighbour_num] = {}
        scores = [knn_fold_results[f'knn_results_fold_{i}'][f'distance_{neighbour_num}']
                  for i in range(d)]
        
        train_summary, test_summary = get_train_test_summaries(scores)
        
        neighbour_perf[neighbour_num]['train'] = train_summary
        neighbour_perf[neighbour_num]['test'] = test_summary
    
    knn_fold_results['neighbour_performance'] = neighbour_perf        
    
    best_fold_test_neighbour_num = max(
        neighbour_perf, 
        key = lambda n: neighbour_perf[n]['test'].mean)
            
    num_neighbours_test = [best_fold_test_neighbour_num]
    
    if center_norm_after_train:
        all_train_embeddings, test_embeddings = reduction.center_dimensions_normalize_embeddings(all_train_embeddings, test_embeddings)            
    
    elif force_dist == ForceDistribution.NORMAL_ALL or force_dist == ForceDistribution.UNIFORM_ALL:
        all_train_embeddings, test_embeddings = reduction.transform_all_embeddings(
            train_embeddings = all_train_embeddings_not_dist,
            test_embeddings = test_embeddings,
            force_dist = force_dist,
            random_state = random_state)
        
    elif force_dist == ForceDistribution.NORMAL_SPLITS:
        all_train_embeddings = reduction.transform_dimensions(
            all_train_embeddings, 
            random_state,
            distribution_type = reduction.ProbabilityDistribution.NORMAL,
            normalize_rows = True)
        test_embeddings = reduction.transform_dimensions(
            test_embeddings, 
            random_state,
            distribution_type = reduction.ProbabilityDistribution.NORMAL,
            normalize_rows = True)        
    
    elif force_dist == ForceDistribution.UNIFORM_SPLITS:
        all_train_embeddings = reduction.transform_dimensions(
            all_train_embeddings, 
            random_state,
            distribution_type = reduction.ProbabilityDistribution.UNIFORM,
            normalize_rows = True)
        test_embeddings = reduction.transform_dimensions(
            test_embeddings, 
            random_state,
            distribution_type = reduction.ProbabilityDistribution.UNIFORM,
            normalize_rows = True)
        
    
    knn_results_test = get_nearest_neighbour_results_graph(
        all_train_embeddings,
        all_train_targets,
        test_embeddings,
        test_targets,
        output_folder,
        num_neighbours_test,
        skhubness_reduction = skhubness_reduction,
        get_class_scores = get_class_scores_test,
        n_jobs = n_jobs,
        get_correct_test_idx = True)
    
    knn_fold_results['knn_results_test'] = knn_results_test
    
    nmslib_wrap.delete_indexes_if_exist()
    
    return knn_fold_results






