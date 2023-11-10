# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 09:33:16 2022

@author: bmgi


performance of various models k-nearest neighbours and hubness


"""



from datetime import datetime

import logging
import numpy as np
from numpy.typing import NDArray

import os

import re

from src.skhubness_custom import nmslib_wrap

from src import knn
from src import sentence_bert as sent_bert

from src.util import naming, save_to_file, summary_stats
from src.hubness import analysis, reduction
from src.hubness.reduction import ForceDistribution, SkhubnessReduction

from tqdm import tqdm
from typing import Dict, List




def get_result_name_suffix(
        center_norm_after_train: bool,
        skhubness_reduction: SkhubnessReduction,
        force_dist: ForceDistribution,
        d_fold: int) -> str:
    suffix = ''
    if d_fold > 0:
        suffix = suffix + f'_{d_fold}_fold'
    if center_norm_after_train:
        suffix = suffix + '_center_norm_after_train'
    if force_dist != ForceDistribution.NONE:
        suffix = suffix + f'_force_{force_dist.value}'
    if skhubness_reduction != SkhubnessReduction.NONE:
        suffix = suffix + f'_reduction_{skhubness_reduction.value}'
    
    return suffix


def get_result_name(
        result_type: str,
        model_name: str, 
        dataset: str,
        center_norm_after_train: bool,
        skhubness_reduction: SkhubnessReduction,
        force_dist: ForceDistribution,
        d_fold: int) -> str:
    base_name = f'{model_name}_{dataset}_{result_type}_results'
    suffix = get_result_name_suffix(
        center_norm_after_train,        
        skhubness_reduction,
        force_dist, 
        d_fold)
    
    return base_name + suffix + '.json'


def get_d_fold_performance_of_models(
        output_folder: str,
        model_names: List[str],
        dataset: str,
        device: str,
        random_state: int,
        center_norm_after_train: bool,
        skhubness_reduction: SkhubnessReduction = SkhubnessReduction.NONE,
        force_dist: ForceDistribution = ForceDistribution.NONE,
        d: int = 5, 
        get_class_scores_fold: bool = False,
        get_class_scores_test: bool = True,
        n_jobs: int = 5) -> None:
    
    embedding_folder_path,results_folder_path = naming.check_folders(output_folder)
    
    subsets = ['train', 'test']
    
    data_dict = {}
    data_dict = knn.get_features_and_targets(dataset, subsets, data_dict)
        
    all_train_targets = data_dict['train'].targets
    test_targets = data_dict['test'].targets
    
    num_neighbours = range(2, 20)
    
    for model_name in model_names:
        results_name = get_result_name(
            'knn',
            model_name, 
            dataset,
            center_norm_after_train = center_norm_after_train,
            skhubness_reduction = skhubness_reduction,
            force_dist = force_dist,
            d_fold = d)
        results_path = os.path.join(results_folder_path, results_name)        
        
        if os.path.exists(results_path):
            continue
        
        logging.info(f'Will make result: {results_path}')
        nmslib_wrap.write_to_log(results_name, output_folder)
        
        
        data_dict = knn.get_all_embeddings(
            output_folder = output_folder,
            model_name = model_name, 
            dataset = dataset,
            subsets = subsets,
            device = device,
            embedding_folder_path = embedding_folder_path,
            data_dict = data_dict)
        
        all_train_embeddings = data_dict['train'].embeddings
        test_embeddings = data_dict['test'].embeddings
        
        knn_fold_results = knn.get_d_fold_results(
            all_train_embeddings,
            all_train_targets,
            test_embeddings, 
            test_targets,
            output_folder = output_folder,
            num_neighbours = num_neighbours,
            random_state = random_state,
            center_norm_after_train = center_norm_after_train,
            skhubness_reduction = skhubness_reduction,
            force_dist = force_dist,
            d = d, 
            get_class_scores_fold = get_class_scores_fold,
            get_class_scores_test = get_class_scores_test,
            n_jobs = n_jobs)
        
        save_to_file.save_as_json(knn_fold_results, save_to = results_path)


def get_d_fold_performance_of_models_from_file(
        output_folder: str,
        model_names_path: str,
        dataset: str,
        device: str,
        random_state: int,
        center_norm_after_train: bool,
        skhubness_reduction: SkhubnessReduction = SkhubnessReduction.NONE,
        force_dist: ForceDistribution = ForceDistribution.NONE,
        d: int = 10, 
        get_class_scores_fold: bool = False,
        get_class_scores_test: bool = True,
        n_jobs: int = 5) -> None:
    
    model_names = get_model_names_from_file(model_names_path)
    
    get_d_fold_performance_of_models(
        output_folder, 
        model_names, 
        dataset, 
        device, 
        random_state,
        center_norm_after_train,
        skhubness_reduction,
        force_dist,
        d,
        get_class_scores_fold,
        get_class_scores_test,
        n_jobs)


def decode_knn_results(
        json_dict: Dict[str, object], 
        knn_result_keys: List[str]) -> Dict[str, Dict[str, knn.KnnScores]]:
    
    decoded_results = {}
    
    for main_key in knn_result_keys:
        decoded_results[main_key] = {}
        knn_results = json_dict[main_key]
        for key in knn_results:
            if 'correct_test_idx' in knn_results[key]:
                correct_test_idx = np.array(knn_results[key]['correct_test_idx'])
            else:
                correct_test_idx = np.array([])
                
            decoded_results[main_key][key] = knn.KnnScores(
                knn_results[key]['train_score'], 
                knn_results[key]['test_score'],
                correct_test_idx)
    
    return decoded_results


def decode_d_shot_results(
        json_dict: Dict[str, object]) -> Dict[str, Dict[str, knn.KnnScores]]:
    
    knn_result_keys = [key for key in json_dict if 'knn_results' in key]
    other_keys = [key for key in json_dict if 'knn_results' not in key]
    
    decoded_results = decode_knn_results(json_dict, knn_result_keys)
    
    for key in other_keys:
        decoded_results[key] = json_dict[key]
    
    return decoded_results


def load_result(result_path: str) -> Dict[str, knn.KnnScores]:
    loaded_results = {}
    
    json_dict = save_to_file.load_json(result_path)
    
    loaded_results = decode_d_shot_results(json_dict)            
    
    return loaded_results


def get_knn_test_results_results_path(
        results_folder_path: str, 
        model_result_file: str) -> Dict[str, knn.KnnScores]:
    
    results_path = os.path.join(results_folder_path, model_result_file)
    results = load_result(results_path)
    
    test_results = results['knn_results_test']
    
    return test_results


def get_knn_test_results(
        output_folder: str, 
        model_result_file: str,
        results_folder: str = '') -> Dict[str, knn.KnnScores]:
    
    if results_folder == '':
        results_folder_path = os.path.join(output_folder, naming.get_results_folder())
    else:
        results_folder_path = os.path.join(output_folder, results_folder)    
    
    test_results = get_knn_test_results_results_path(results_folder_path, model_result_file)
    
    return test_results
   

def get_model_names_from_file(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        text = f.read() 
        
    model_names = text.split(',')
    
    return model_names


def get_test_results_with_standard_keys(
        result: Dict[str, knn.KnnScores]) -> Dict[str, knn.KnnScores]:
    result_new_key = {re.sub('distance_\d+_', '', key): result[key] 
                       for key in result}
    result_new_key = {re.sub('distance_\d+$', 'all', key): result_new_key[key] 
                       for key in result_new_key}
    
    return result_new_key


def knn_score_diff(score_from: knn.KnnScores, score_to: knn.KnnScores) -> knn.KnnScores:
    return knn.KnnScores(
        train_score = score_to.train_score - score_from.train_score, 
        test_score = score_to.test_score - score_from.test_score)


def get_accuracy_differences(
        output_folder: str,
        model_name_root: str,
        dataset: str,
        suffix_from: str,
        suffix_to: str) -> Dict[str, float]:
    
    results_dir_path = os.path.join(output_folder, naming.get_results_folder())    
    files_in_results = [file 
                        for file in os.listdir(results_dir_path)
                        if file.endswith('.json')]
    
    model_result_files_from = [
        file 
        for file in files_in_results
        if re.match(f'{model_name_root}.*_{dataset}_knn_results{suffix_from}.json', file)]
    
    model_result_files_to = [
        file 
        for file in files_in_results
        if re.match(f'{model_name_root}.*_{dataset}_knn_results{suffix_to}.json', file)]
    
    differences = {}
    
    for model_result_file_from in model_result_files_from:        
        common_name = model_result_file_from.split('_knn_results')[0]
        matching_to_file = [file 
                            for file in model_result_files_to 
                            if re.match(f'{common_name}_knn_results.*', file)]
        
        if len(matching_to_file) < 1:
            raise ValueError(f'No matching to file for: {model_result_file_from}')
               
        t_results_from = get_knn_test_results(output_folder, model_result_file_from)
        t_results_to = get_knn_test_results(output_folder, matching_to_file[0])
        t_results_from_std = get_test_results_with_standard_keys(t_results_from)
        t_results_to_std = get_test_results_with_standard_keys(t_results_to)
        test_diff = {
            key: knn_score_diff(t_results_from_std[key], t_results_to_std[key])
            for key in t_results_from_std}
        diff_key = common_name.replace(model_name_root + '_', '')
        
        differences[diff_key] = test_diff
    
    return differences


def get_suffix_difference_summary(
        output_folder: str,
        model_name_roots: List[str],
        dataset: str,
        suffix_from: str,
        suffix_to: str) -> float:
    
    diff_summary = {}
    
    for model_name_root in model_name_roots:
        diff_summary[model_name_root] = {}
        differences = get_accuracy_differences(
            output_folder,
            model_name_root,
            dataset,
            suffix_from,
            suffix_to)
        
        diff_summary[model_name_root]['differences'] = differences
        
        diff_all = [differences[key]['all'] for key in differences]        
        train_diff_summary, test_diff_summary = knn.get_train_test_summaries(diff_all)
        
        diff_summary[model_name_root]['train_diff_summary'] = train_diff_summary
        diff_summary[model_name_root]['test_diff_summary'] = test_diff_summary
        
    return diff_summary


def save_suffix_difference_summary(
        output_folder: str,
        model_name_roots: List[str],
        dataset: str,
        suffix_from: str,
        suffix_to: str) -> None:
    
    diff_summary = get_suffix_difference_summary(
            output_folder = output_folder,
            model_name_roots = model_name_roots,
            dataset = dataset,
            suffix_from = suffix_from,
            suffix_to = suffix_to)
    
    summary_name = f'{dataset}_{datetime.now().strftime("%Y-%m-%d_%H-%M")}_knn_summary_diff{suffix_from}_to{suffix_to}.json'
    summary_path = os.path.join(output_folder, naming.get_results_folder(),summary_name)   
    
    save_to_file.save_as_json(diff_summary, save_to = summary_path)


def get_summary_stats_of_knn_scores(
        results: List[Dict[str, knn.KnnScores]]) -> Dict[str, Dict[str, summary_stats.SummaryStats]]:
    summary_dict = {}
    
    new_key_results = [get_test_results_with_standard_keys(result)
                       for result in results]    
    
    for key in new_key_results[0]:
        summary_dict[key] = {}
        train_test_scores = [result[key] for result in new_key_results]
        
        train_summary, test_summary = knn.get_train_test_summaries(train_test_scores)
        
        summary_dict[key]['train'] = train_summary
        summary_dict[key]['test'] = test_summary
    
    return summary_dict


def get_summary_stats_knn_results(
        output_folder: str,
        model_name_roots: List[str],
        dataset: str,
        center_norm_after_train: bool,
        skhubness_reduction: SkhubnessReduction,
        force_dist: ForceDistribution,
        d_fold: int,
        results_folder: str = '') -> Dict[str, Dict[str, Dict[str, summary_stats.SummaryStats]]]:
    
    model_summaries = {}
    
    if results_folder == '':
        results_dir_path = os.path.join(output_folder, naming.get_results_folder())
    else:
        results_dir_path = os.path.join(output_folder, results_folder)
        
    files_in_results = [file 
                        for file in os.listdir(results_dir_path)
                        if file.endswith('.json')]
    
    suffix = get_result_name_suffix(
        center_norm_after_train,
        skhubness_reduction,
        force_dist, 
        d_fold)
    
    for model_name_root in model_name_roots:
        model_result_files = [
            file 
            for file in files_in_results
            if re.match(f'{model_name_root}.*_{dataset}_knn_results{suffix}.json', file)]
        
        model_test_results = []
        
        for model_result_file in model_result_files:
            test_results = get_knn_test_results(output_folder, model_result_file, results_folder)
            
            model_test_results.append(test_results)
        
        model_summary = get_summary_stats_of_knn_scores(model_test_results)
               
        model_summaries[model_name_root] = model_summary
    
    return model_summaries


def get_and_save_summary_stats(
        output_folder: str,
        model_name_roots: List[str],
        dataset: str,
        center_norm_after_train: bool,
        skhubness_reduction: SkhubnessReduction,
        force_dist: ForceDistribution,
        d_fold: int) -> None:
    
    model_summaries = get_summary_stats_knn_results(
        output_folder,
        model_name_roots,
        dataset,
        center_norm_after_train,
        skhubness_reduction,
        force_dist,
        d_fold)
    
    suffix = get_result_name_suffix(
        center_norm_after_train,
        skhubness_reduction,
        force_dist, 
        d_fold)      
    
    summary_name = f'{dataset}_{datetime.now().strftime("%Y-%m-%d_%H-%M")}_knn_summary{suffix}.json'
    summary_path = os.path.join(output_folder, naming.get_results_folder(),summary_name)   
    
    save_to_file.save_as_json(model_summaries, save_to = summary_path)


def get_and_save_summary_stats_default_model_name_root_d_fold(
        output_folder: str,
        dataset: str,
        center_norm_after_train: bool,
        force_dist: ForceDistribution,
        d_fold: int) -> None:
    
    model_name_roots = naming.get_model_name_roots_from_knn_results(output_folder, dataset)
    
    get_and_save_summary_stats(
        output_folder, 
        model_name_roots, 
        dataset,
        center_norm_after_train,
        force_dist,
        d_fold)


def save_suffix_difference_summary_stats_default_model_name_root_d_fold(
        output_folder: str,
        dataset: str,
        suffix_from: str,
        suffix_to: str) -> None:
    
    results_dir_path = os.path.join(output_folder, naming.get_results_folder())    
    files_in_results = [file 
                        for file in os.listdir(results_dir_path)
                        if re.match(f'.*_{dataset}_knn_results_.*\.json', file)]
    
    model_name_roots = list(set([re.split('_seed\d+', file)[0] 
                                 for file in files_in_results]))
    
    save_suffix_difference_summary(
        output_folder, 
        model_name_roots, 
        dataset,
        suffix_from,
        suffix_to)


def get_d_fold_performance_of_models_from_embedding_files(
        output_folder: str,
        dataset: str,
        device: str,
        random_state: int,
        center_norm_after_train: bool,
        skhubness_reduction: SkhubnessReduction,
        force_dist: ForceDistribution,
        d: int = 10,
        n_jobs: int = 5) -> None:
    
    embedding_model_names = naming.get_model_names_from_embeddings(output_folder, dataset)
    
    get_d_fold_performance_of_models(
        output_folder = output_folder,
        model_names = embedding_model_names,
        dataset = dataset,
        device = device,
        random_state = random_state,
        center_norm_after_train = center_norm_after_train,
        skhubness_reduction = skhubness_reduction,
        force_dist = force_dist,
        d = d, 
        get_class_scores_fold = False,
        get_class_scores_test = True,
        n_jobs = n_jobs)
    

def get_d_fold_performance_of_models_from_model_folder(
        output_folder: str,
        model_prefix: str,
        dataset: str,
        device: str,
        random_state: int,
        center_norm_after_train: bool,
        skhubness_reduction: SkhubnessReduction,
        force_dist: ForceDistribution,
        d: int = 10,
        n_jobs: int = 5) -> None:
    
    model_names = naming.get_model_names_from_model_folder(output_folder, model_prefix)
    
    get_d_fold_performance_of_models(
        output_folder = output_folder,
        model_names = model_names,
        dataset = dataset,
        device = device,
        random_state = random_state,
        center_norm_after_train = center_norm_after_train,
        skhubness_reduction = skhubness_reduction,
        force_dist = force_dist, 
        d = d, 
        get_class_scores_fold = False,
        get_class_scores_test = True,
        n_jobs = n_jobs)


def get_graphs_and_hubness_scores(
        k: int,
        neighbour_num: int, 
        train_embeddings: NDArray,
        test_embeddings: NDArray,
        output_folder: str):
    train_graph, test_graph = knn.get_knn_graph(
        neighbour_num = neighbour_num,
        train_features = train_embeddings,        
        test_features = test_embeddings,
        output_folder = output_folder,
        knn_metric = 'euclidean',
        knn_mode = 'distance')
    
    train_scores = analysis.get_k_hubness_scores(train_graph, k = k)
    test_scores = analysis.get_k_hubness_scores(test_graph, k = k)
    
    return train_graph, test_graph, train_scores, test_scores    


def get_hubness_from_embedding_files(
        output_folder: str,
        dataset: str,
        random_state: int,
        skhubness_reductions: List[SkhubnessReduction],
        force_dists: List[ForceDistribution]) -> None:
    
    k = 10    
    embedding_folder_path,results_folder_path = naming.check_folders(output_folder)
    
    embedding_model_names = naming.get_model_names_from_embeddings(output_folder, dataset)
    
    for model_name in tqdm(embedding_model_names):
        results_name = get_result_name(
            'hubness',
            model_name, 
            dataset,
            center_norm_after_train = False,
            skhubness_reduction = SkhubnessReduction.NONE,
            force_dist = ForceDistribution.NONE,
            d_fold = 0)
        results_path = os.path.join(results_folder_path, results_name)
        
        if os.path.exists(results_path):
            continue
        
        results = {}
        
        train_embeddings = sent_bert.load_subset_embeddings(
                model_name = model_name, 
                dataset = dataset, 
                subset = 'train',
                embedding_folder_path = embedding_folder_path)
        test_embeddings = sent_bert.load_subset_embeddings(
                model_name = model_name, 
                dataset = dataset, 
                subset = 'test',
                embedding_folder_path = embedding_folder_path)
        
        neighbour_num = 20
        
        if ForceDistribution.NORMAL_ALL in force_dists:
            train_embeddings_norm_dist, test_embeddings_norm_dist = reduction.transform_all_embeddings(
                train_embeddings = train_embeddings,
                test_embeddings = test_embeddings,
                force_dist = ForceDistribution.NORMAL_ALL,
                random_state = random_state)
            
            norm_dist_res = get_graphs_and_hubness_scores(
                k,
                neighbour_num = neighbour_num, 
                train_embeddings = train_embeddings_norm_dist,
                test_embeddings = test_embeddings_norm_dist,
                output_folder = output_folder)            
            train_graph_n_dist, test_graph_n_dist, train_scores_n_dist, test_scores_n_dist = norm_dist_res
            
            n_all_key = ForceDistribution.NORMAL_ALL.value
            results[n_all_key] = {} 
            results[n_all_key]['train'] = train_scores_n_dist
            results[n_all_key]['test'] = test_scores_n_dist
            
        if ForceDistribution.UNIFORM_ALL in force_dists:
            train_embeddings_uni_dist, test_embeddings_uni_dist = reduction.transform_all_embeddings(
                train_embeddings = train_embeddings,
                test_embeddings = test_embeddings,
                force_dist = ForceDistribution.UNIFORM_ALL,
                random_state = random_state)
            
            uni_dist_res = get_graphs_and_hubness_scores(
                k,
                neighbour_num = neighbour_num, 
                train_embeddings = train_embeddings_uni_dist,
                test_embeddings = test_embeddings_uni_dist,
                output_folder = output_folder)            
            train_graph_u_dist, test_graph_u_dist, train_scores_u_dist, test_scores_u_dist = uni_dist_res
            
            u_all_key = ForceDistribution.UNIFORM_ALL.value
            results[u_all_key] = {} 
            results[u_all_key]['train'] = train_scores_u_dist
            results[u_all_key]['test'] = test_scores_u_dist
        
        if ForceDistribution.NORMAL_SPLITS in force_dists:
            train_embeddings_n_sp = reduction.transform_dimensions(
                train_embeddings, 
                random_state,
                distribution_type = reduction.ProbabilityDistribution.NORMAL,
                normalize_rows = True)
            test_embeddings_n_sp = reduction.transform_dimensions(
                test_embeddings, 
                random_state,
                distribution_type = reduction.ProbabilityDistribution.NORMAL,
                normalize_rows = True)
            
            n_sp_res = get_graphs_and_hubness_scores(
                k,
                neighbour_num = neighbour_num, 
                train_embeddings = train_embeddings_n_sp,
                test_embeddings = test_embeddings_n_sp,
                output_folder = output_folder)            
            train_graph_n_sp, test_graph_n_sp, train_scores_n_sp, test_scores_n_sp = n_sp_res
            
            n_sp_key = ForceDistribution.NORMAL_SPLITS.value
            results[n_sp_key] = {} 
            results[n_sp_key]['train'] = train_scores_n_sp
            results[n_sp_key]['test'] = test_scores_n_sp
        
        if ForceDistribution.UNIFORM_SPLITS in force_dists:
            train_embeddings_u_sp = reduction.transform_dimensions(
                train_embeddings, 
                random_state,
                distribution_type = reduction.ProbabilityDistribution.UNIFORM,
                normalize_rows = True)
            test_embeddings_u_sp = reduction.transform_dimensions(
                test_embeddings, 
                random_state,
                distribution_type = reduction.ProbabilityDistribution.UNIFORM,
                normalize_rows = True)
            
            u_sp_res = get_graphs_and_hubness_scores(
                k,
                neighbour_num = neighbour_num, 
                train_embeddings = train_embeddings_u_sp,
                test_embeddings = test_embeddings_u_sp,
                output_folder = output_folder)            
            train_graph_u_sp, test_graph_u_sp, train_scores_u_sp, test_scores_u_sp = u_sp_res
            
            u_sp_key = ForceDistribution.UNIFORM_SPLITS.value
            results[u_sp_key] = {} 
            results[u_sp_key]['train'] = train_scores_u_sp
            results[u_sp_key]['test'] = test_scores_u_sp
        
        
        if SkhubnessReduction.MUTUAL_PROXIMITY in skhubness_reductions:
            neighbour_num += 100        
        
        base_res = get_graphs_and_hubness_scores(
            k,
            neighbour_num = neighbour_num, 
            train_embeddings = train_embeddings,
            test_embeddings = test_embeddings,
            output_folder = output_folder)            
        train_graph, test_graph, train_scores, test_scores = base_res
        
        base_key = 'base'
        results[base_key] = {}        
        results[base_key]['train'] = train_scores
        results[base_key]['test'] = test_scores
        
        if SkhubnessReduction.MUTUAL_PROXIMITY in skhubness_reductions:
            train_graph_mp, test_graph_mp = reduction.get_mutual_proximity_graphs(
                train_graph, test_graph)
            
            train_graph_mp, test_graph_mp = knn.check_graphs_for_nan_and_inf(train_graph_mp, test_graph_mp)
            
            train_scores_mp = analysis.get_k_hubness_scores(train_graph_mp, k = k)
            test_scores_mp = analysis.get_k_hubness_scores(test_graph_mp, k = k)
            
            mp_key = SkhubnessReduction.MUTUAL_PROXIMITY.value
            results[mp_key] = {} 
            results[mp_key]['train'] = train_scores_mp
            results[mp_key]['test'] = test_scores_mp
        
        if SkhubnessReduction.MUTUAL_PROXIMITY in skhubness_reductions and ForceDistribution.NORMAL_ALL in force_dists:
            train_graph_n_mp, test_graph_n_mp = reduction.get_mutual_proximity_graphs(
                train_graph_n_dist, test_graph_n_dist)
            
            train_graph_n_mp, test_graph_n_mp = knn.check_graphs_for_nan_and_inf(train_graph_n_mp, test_graph_n_mp)
            
            train_scores_n_mp = analysis.get_k_hubness_scores(train_graph_n_mp, k = k)
            test_scores_n_mp = analysis.get_k_hubness_scores(test_graph_n_mp, k = k)
                                    
            force_norm_mp_key = ForceDistribution.NORMAL_ALL.value + '_' + SkhubnessReduction.MUTUAL_PROXIMITY.value
            results[force_norm_mp_key] = {} 
            results[force_norm_mp_key]['train'] = train_scores_n_mp
            results[force_norm_mp_key]['test'] = test_scores_n_mp
        
        if SkhubnessReduction.LOCAL_SCALING in skhubness_reductions:
            train_graph_ls, test_graph_ls = reduction.get_local_scaling_graphs(
                train_graph, test_graph)
            
            train_graph_ls, test_graph_ls = knn.check_graphs_for_nan_and_inf(train_graph_ls, test_graph_ls)
            
            train_scores_ls = analysis.get_k_hubness_scores(train_graph_ls, k = k)
            test_scores_ls = analysis.get_k_hubness_scores(test_graph_ls, k = k)
            
            ls_key = SkhubnessReduction.LOCAL_SCALING.value
            results[ls_key] = {} 
            results[ls_key]['train'] = train_scores_ls
            results[ls_key]['test'] = test_scores_ls
            
        save_to_file.save_as_json(results, save_to = results_path)
        
        nmslib_wrap.delete_indexes_if_exist()
        





