# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 09:07:44 2022

@author: bmgi


Collecting the final accuracy and hubness results


"""


import logging
import math
import matplotlib.pyplot as plt 
import numpy as np
from numpy.typing import NDArray

import os
import pandas as pd

import re

from scipy.stats.stats import pearsonr, spearmanr

import seaborn as sns
from src.hubness import analysis
from src.hubness.reduction import ForceDistribution, SkhubnessReduction
from src.performance import saving_results
from src.util import naming, save_to_file, summary_stats, vectors

from tqdm import tqdm
from typing import Dict, List, Tuple



def get_hubness_reductions() -> List[str]:
    return ['base', 'normal_all', 'uniform_all', 'mutual_proximity', 'normal_all_mutual_proximity', 'local_scaling']


def get_subsets() -> List[str]:
    return ['train', 'test']


def get_hubness_measures() -> List[str]:
    return ['k_skewness', 'robinhood']


def convert_hubness_reduction_name(
        hubness_reduction_name: str) -> Tuple[ForceDistribution, SkhubnessReduction]:
    if hubness_reduction_name == 'base':
        return ForceDistribution.NONE, SkhubnessReduction.NONE
    
    if hubness_reduction_name == 'normal_all_mutual_proximity':
        return ForceDistribution.NORMAL_ALL, SkhubnessReduction.MUTUAL_PROXIMITY
    
    if '_all' in hubness_reduction_name:
        return ForceDistribution(hubness_reduction_name), SkhubnessReduction.NONE
    
    return ForceDistribution.NONE, SkhubnessReduction(hubness_reduction_name)


def get_reduction_short_name(reduction: str):
    if reduction == 'base':
        return reduction 
    if reduction == 'normal_all':
        return 'f-norm'
    if reduction == 'uniform_all':
        return 'f-uniform'
    if reduction == 'mutual_proximity':
        return 'MP'
    if reduction == 'local_scaling':
        return 'local scaling'
    if reduction == 'normal_all_mutual_proximity':
        return 'f-norm + MP'


def get_hubness_results(
        model_name_roots: List[str],
        output_folder: str,        
        dataset: str,
        prefix: str = '',
        results_folder: str = '') -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]]:
    
    if results_folder == '':
        results_path = os.path.join(output_folder, naming.get_results_folder())
    else:
        results_path = os.path.join(output_folder, results_folder)
    
    hub_files = [file 
                 for file in os.listdir(results_path)
                 if re.match(f'{prefix}.*_{dataset}_hubness_results.json', file)]
    
    hubness_results = {root: {} for root in model_name_roots}
    for file in hub_files:
        model_name = file.split(f'_{dataset}_')[0]
        model_name_root, seed = model_name.split('_seed')
        seed = int(seed)        
        hubness_results[model_name_root][seed] = {}
        seed_dict = hubness_results[model_name_root][seed]
        
        hub_reductions = get_hubness_reductions()
        subsets = get_subsets()
        measures = get_hubness_measures()
                
        file_path = os.path.join(results_path, file)
        json_dict = save_to_file.load_json(file_path)
        
        for red in hub_reductions:
            if red in json_dict:
                seed_dict[red] = {}
                for subset in subsets:
                    seed_dict[red][subset] = {}
                    for measure in measures:
                        seed_dict[red][subset][measure] = json_dict[red][subset][measure]
                    
    return hubness_results


def get_reduction_summaries(
        model_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], 
        red: str,
        subsets: List[str], 
        measures: List[str]) -> Dict[str, Dict[str, summary_stats.SummaryStats]]:
    red_summaries = {}
    for subset in subsets:
        red_summaries[subset] = {}
        for measure in measures:
            results = [model_results[seed][red][subset][measure]
                       for seed in model_results]
            res_summary = summary_stats.SummaryStats(results)
            
            red_summaries[subset][measure] = res_summary
    
    return red_summaries


def get_hubness_summaries(
        hubness_results: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]) -> Dict[str, Dict[str, Dict[str, Dict[str, summary_stats.SummaryStats]]]]:
    model_summaries = {}
    for model_name_root in hubness_results:
        model_summaries[model_name_root] = {}
        model_results = hubness_results[model_name_root]
        
        for red in get_hubness_reductions():
            
            if red not in next(iter(model_results.values())):
                continue                
            
            reduction_summaries = get_reduction_summaries(
                model_results,
                red,
                get_subsets(),
                get_hubness_measures())
            
            model_summaries[model_name_root][red] = reduction_summaries
    
    return model_summaries


def get_all_knn_summaries(
        model_name_roots: List[str],
        output_folder: str, 
        dataset: str,
        reductions: List[str],
        results_folder: str = '') -> Dict[str, Dict[str, Dict[str, summary_stats.SummaryStats]]]:    
    
    knn_summaries = {root: {} for root in model_name_roots}
    
    for red in reductions:
        force_dist, skhubness_reduction = convert_hubness_reduction_name(red)
        
        red_summary = saving_results.get_summary_stats_knn_results(
            output_folder = output_folder,
            model_name_roots = model_name_roots,
            dataset = dataset,
            center_norm_after_train = False,
            skhubness_reduction = skhubness_reduction,
            force_dist = force_dist,
            d_fold = 10,
            results_folder = results_folder)
        
        for root in red_summary:
            knn_summaries[root][red] = red_summary[root]['all']
    
    return knn_summaries


def unpack_summary_stats(summary: summary_stats.SummaryStats):
    return [summary.min_value, summary.max_value, summary.mean]
    

def get_multi_index_for_models(
        model_names: List[str],
        hubness_reductions: List[str],
        use_seeds: bool):
    all_base_models = ['microsoft-MiniLM-L12-H384-uncased','distilroberta-base', 'microsoft-mpnet-base']
    relevant_base_models = [base_model 
                            for base_model in all_base_models
                            if any([base_model in model_name 
                                   for model_name in model_names])]
    
    parameters = [
        relevant_base_models,
        ['cos', 'cos_dist', 'euclidean'],
        ['none', 'c', 'n', 'c, n', 'z']
        ]
    
    names = ['base model', 'distance', 'training normalisation']
    
    if hubness_reductions:
        parameters.append(hubness_reductions)
        names.append('hubness reduction')
    if use_seeds:
        seeds = list(set([naming.get_seed_from_model_name(name)
                 for name in model_names]))
        seeds.sort(key = lambda s: int(s))
        parameters.append(seeds)
        names.append('seeds')        
    

    multi_index = pd.MultiIndex.from_product(parameters, names = names)
    
    return multi_index


def get_multi_index_dataframe_summary(
        model_names: List[str], 
        subsets: List[str],
        hubness_reductions: List[str]):
    multi_index = get_multi_index_for_models(
        model_names, 
        hubness_reductions = hubness_reductions,
        use_seeds = False)
    
    stats = ['min', 'max', 'mean']
    measures = ['acc', 'skew', 'rh']
    columns = [t + ' ' + m + ' ' + s 
               for m in measures 
               for t in subsets 
               for s in stats]    
    
    df = pd.DataFrame(index = multi_index, columns = columns)
    
    return df


def get_multi_index_dataframe_pairs(
        model_names: List[str], 
        subsets: List[str],
        hubness_reductions: List[str]):
    multi_index = get_multi_index_for_models(
        model_names, 
        hubness_reductions = [],
        use_seeds = True)
        
    measures = ['acc', 'skew', 'rh']
    columns = [m + ' ' + red + ' ' + s 
               for m in measures 
               for red in hubness_reductions
               for s in subsets]    
    
    df = pd.DataFrame(index = multi_index, columns = columns)
    
    return df



def build_overview_df_from_results(
        output_folder: str, 
        dataset: str,
        subsets: List[str] = ['test'],
        prefix: str = '',
        results_folder: str = '',
        use_short_names: bool = False,
        use_error_rate: bool = False) -> pd.DataFrame:
    model_name_roots = naming.get_model_name_roots_from_knn_results(
        output_folder, 
        dataset,
        prefix,
        results_folder)
    
    hubness_results = get_hubness_results(
        model_name_roots, output_folder, dataset, prefix, results_folder)
    hubness_summaries = get_hubness_summaries(hubness_results)
    
    reductions = [r for r in get_hubness_reductions() 
                  if r in next(iter(hubness_summaries.values()))]
    knn_summaries = get_all_knn_summaries(
        model_name_roots, output_folder, dataset, reductions, results_folder)
    
    reduction_names = [get_reduction_short_name(r) for r in reductions]
    df = get_multi_index_dataframe_summary(model_name_roots, subsets, reduction_names)    
    
    for root in tqdm(model_name_roots):
        base_model = naming.get_base_model_from_model_name(root)
        distance = naming.get_distance_from_model_name(root)
        train_norm = naming.get_train_norm_from_model_name(root)
        
        for red in reductions:
            red_name = get_reduction_short_name(red)
            result_list = []
            
            for subset in subsets:
                acc_summary = knn_summaries[root][red][subset]
                summary = unpack_summary_stats(acc_summary)
                if use_error_rate:
                    summary = [1 - s for s in summary]
                
                result_list.extend(summary)
            
            for subset in subsets:
                skew_summary = hubness_summaries[root][red][subset]['k_skewness']
                result_list.extend(unpack_summary_stats(skew_summary))
                
            for subset in subsets:
                rh_summary = hubness_summaries[root][red][subset]['robinhood']
                result_list.extend(unpack_summary_stats(rh_summary))
            
            df.loc[base_model, distance, train_norm, red_name] = result_list
        
    return df


def build_table(data, row_labels, col_width=3.0, row_height=0.625, 
                font_size_headers = 14, font_size_cells = 16,
                header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                bbox=[0, 0, 1, 1], header_columns=0,
                ax=None, **kwargs):
        
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, rowLabels = row_labels, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size_headers)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
            cell.set_fontsize(font_size_cells)
    return ax.get_figure(), ax


def build_and_save_overview_table_from_results(
        output_folder: str, 
        dataset: str,
        save_as: str,
        subsets: List[str] = ['test'],
        prefix: str = '',
        results_folder: str = '',
        use_error_rate: bool = False) -> None:
    
    overview_df = build_overview_df_from_results(
        output_folder, 
        dataset, 
        subsets,
        prefix,
        results_folder,
        use_error_rate = use_error_rate)
    
    if len(subsets) == 1:
        cols = overview_df.columns
        if use_error_rate:            
            col_rename_map = {col: col.replace(f'{subsets[0]} ', '').replace('acc', 'error')
                              for col in cols}
        else:
            col_rename_map = {col: col.replace(f'{subsets[0]} ', '')
                              for col in cols}
        overview_df = overview_df.rename(columns = col_rename_map)
        col_width = 2
    else:
        col_width = 2.5
        
    
    rounded_overview = overview_df.astype(float).round(decimals = 4)

    index = rounded_overview.index
    row_names = [' '.join(idx) for idx in index]

    fig,ax = build_table(
        rounded_overview, 
        row_names,
        header_columns=0, 
        col_width=col_width)
    
    if save_as != '':
        save_to_file.save_pdf_fig(save_as, output_folder)


def result_suffix_to_hubness_reduction_name(suffix):
    if suffix == '.json':
        return 'base'
    
    return suffix.replace('.json', '').replace('_', '', 1).replace('force_', '').replace('reduction_', '')


def get_hubness_reductions_from_knn_results_results_path(
        results_dir_path: str,
        dataset: str,
        prefix: str = '') -> List[str]:
    files_in_results = [file 
                        for file in os.listdir(results_dir_path)
                        if re.match(f'{prefix}.*_{dataset}_knn_results_.*\.json', file)]
    
    reduction_suffixes = list(set([re.split('_fold', file)[1] 
                                 for file in files_in_results]))    
    
    reduction_names = [result_suffix_to_hubness_reduction_name(s) 
                       for s in reduction_suffixes]
    # To always get them in this order
    reductions = [h for h in get_hubness_reductions() if h in reduction_names]
    
    return reductions


def get_hubness_reductions_from_knn_results(
        output_folder: str,
        dataset: str,
        prefix: str = '',
        results_folder: str = '') -> List[str]:
    
    if results_folder == '':
        results_dir_path = os.path.join(output_folder, naming.get_results_folder())
    else:
        results_dir_path = os.path.join(output_folder, results_folder)    
    
    return get_hubness_reductions_from_knn_results_results_path(
        results_dir_path,
        dataset,
        prefix)    


def get_data_for_paired_comparison(
        output_folder: str, 
        dataset: str,
        reductions: List['str'] = [],
        prefix: str = '',
        subsets: List[str] = ['test'],
        results_folder: str = '') -> Tuple[pd.DataFrame, Dict[str, NDArray]]:
    
    model_names = naming.get_model_names_from_knn_results(
        output_folder, 
        dataset,
        prefix,
        results_folder)
    
    if results_folder == '':
        results_dir_path = os.path.join(output_folder, naming.get_results_folder())
    else:
        results_dir_path = os.path.join(output_folder, results_folder)
    files_in_results = [file 
                        for file in os.listdir(results_dir_path)
                        if file.endswith('.json')]
    
    center_norm_after_train = False
    d_fold = 10       
    
    if not reductions:
        reductions = get_hubness_reductions_from_knn_results(
            output_folder, dataset, prefix, results_folder)
    
    pair_correct_idx_dict = {}
    pair_df = get_multi_index_dataframe_pairs(
        model_names, subsets = subsets, hubness_reductions = reductions)
    
    for model_name in tqdm(model_names):
        base_model = naming.get_base_model_from_model_name(model_name)
        distance = naming.get_distance_from_model_name(model_name)
        train_norm = naming.get_train_norm_from_model_name(model_name)
        seed = naming.get_seed_from_model_name(model_name)
        
        
        hub_files = [file 
                     for file in os.listdir(results_dir_path)
                     if re.match(f'{model_name}_{dataset}_hubness_results.json', file)]
        if len(hub_files) != 1:
            raise ValueError(f'Found: {len(hub_files)} with model_name: {model_name} and dataset: {dataset}')
        
        hub_file = hub_files[0]
        
        hub_file_path = os.path.join(results_dir_path, hub_file)
        hub_json_dict = save_to_file.load_json(hub_file_path)
        
        
        for red in reductions: 
            force_dist, skhubness_reduction = convert_hubness_reduction_name(red)
            
            suffix = saving_results.get_result_name_suffix(
                center_norm_after_train,
                skhubness_reduction,
                force_dist, 
                d_fold)
            
            model_result_files = [
                file 
                for file in files_in_results
                if re.match(f'{model_name}_{dataset}_knn_results{suffix}.json', file)]
            
            if len(model_result_files) != 1:
                raise ValueError(f'Found: {len(model_result_files)} with model_name: {model_name}, dataset: {dataset} and suffix: {suffix}')
            
            model_result_file = model_result_files[0]
            
            test_results = saving_results.get_knn_test_results(
                output_folder, model_result_file, results_folder)
            test_results = saving_results.get_test_results_with_standard_keys(
                test_results)
            
            pair_correct_idx_dict[f'{base_model} {distance} {train_norm} {seed} {red}'] = test_results['all'].correct_test_idx
            
            for subset in subsets:
                acc = test_results['all'].train_score if subset == 'train' else test_results['all'].test_score 
                a_col = f'acc {red} {subset}'
                pair_df.loc[base_model, distance, train_norm, seed][a_col] = acc
                
                s_col = f'skew {red} {subset}'
                pair_df.loc[base_model, distance, train_norm, seed][s_col] = hub_json_dict[red][subset]['k_skewness']
                
                r_col = f'rh {red} {subset}'
                pair_df.loc[base_model, distance, train_norm, seed][r_col] = hub_json_dict[red][subset]['robinhood']
    
    return pair_df, pair_correct_idx_dict


def count_cases_of_increased_measure(
        pair_df: pd.DataFrame,
        reduction_from: str,
        reduction_to: str,
        measure: str,
        base_model: str = '', 
        dist_measure: str = '', 
        train_norm: str = '',
        subset: str = 'test'):
    
    if base_model != '':
        pair_df = pair_df.xs(base_model, drop_level = False)
    if dist_measure != '':
        pair_df = pair_df.xs(dist_measure, level = 1, drop_level = False)
    if train_norm != '':
        pair_df = pair_df.xs(train_norm, level = 2, drop_level = False)
    
    base_values = pair_df[f'{measure} {reduction_from} {subset}'].values
    
    difference = pair_df[f'{measure} {reduction_to} {subset}'].values - base_values
    total = base_values.shape[0]
    
    return (difference > 0).sum(), total


def count_cases_of_increased_accuracy(
        pair_df: pd.DataFrame,
        reduction_from: str,
        reduction_to: str,
        base_model: str = '', 
        dist_measure: str = '', 
        train_norm: str = '',
        subset: str = 'test'):
    
    return count_cases_of_increased_measure(
            pair_df = pair_df,
            reduction_from = reduction_from,
            reduction_to = reduction_to,
            measure = 'acc',
            base_model = base_model, 
            dist_measure = dist_measure, 
            train_norm = train_norm,
            subset = subset)


def build_strict_change_df(
        pair_df: pd.DataFrame,
        change_measure: str,
        subset: str = 'test',
        increase: bool = True):
    columns = pair_df.columns
    reductions = list(set([col.replace(f'{change_measure} ', '').replace(f' {subset}', '')
                  for col in columns
                  if f'{change_measure} ' in col and subset in col]))
    
    reductions_ordered = [red for red in get_hubness_reductions() 
                          if red in reductions]
    
    accuracy_df = pd.DataFrame(columns = reductions_ordered, index = reductions_ordered)
    
    for r_from in reductions_ordered:
        for r_to in reductions_ordered:
            if increase:
                count, total = count_cases_of_increased_measure(
                    pair_df,
                    r_from,
                    r_to,
                    measure = change_measure,
                    subset = subset)
            else:
                count, total = count_cases_of_increased_measure(
                    pair_df,
                    r_to,
                    r_from,
                    measure = change_measure,
                    subset = subset)
            
            accuracy_df.loc[r_from, r_to] = count
    
    return accuracy_df


def build_increased_accuracy_df(
        pair_df: pd.DataFrame,
        subset: str = 'test'):
    
    return build_strict_change_df(
        pair_df = pair_df,
        change_measure = 'acc',
        subset = subset,
        increase = True)    


def build_and_save_increased_acc_decreased_hubness_table(
        output_folder: str,
        dataset: str,
        subset: str,
        save_folder: str = 'output/figures',
        results_folder: str = '') -> None:
    
    pair_df, _ = get_data_for_paired_comparison(
        output_folder = output_folder,
        dataset = dataset,
        subsets = [subset],
        results_folder = results_folder)
    
    accuracy_df = build_increased_accuracy_df(pair_df, subset)
    
    fig,ax = build_table(
        accuracy_df, 
        accuracy_df.index,
        header_columns=0, 
        col_width=5.0)
    
    save_to_file.save_pdf_fig(
        f'{dataset}_accuracy_increases_{subset}', save_folder)
    
    
    rh_reduction_df = build_strict_change_df(
        pair_df,
        change_measure = 'rh',
        subset = subset,
        increase = False)
    
    fig,ax = build_table(
        rh_reduction_df, 
        rh_reduction_df.index,
        header_columns=0, 
        col_width=5.0)
    
    save_to_file.save_pdf_fig(
        f'{dataset}_rh_decreases_{subset}', save_folder)
    
    
    skew_reduction_df = build_strict_change_df(
        pair_df,
        change_measure = 'skew',
        subset = subset,
        increase = False)
    
    fig,ax = build_table(
        skew_reduction_df, 
        skew_reduction_df.index,
        header_columns=0, 
        col_width=5.0)
    
    save_to_file.save_pdf_fig(
        f'{dataset}_skew_decreases_{subset}', save_folder)  


def count_cases_of_increased_accuracy_compared_to_base(
        pair_df: pd.DataFrame,
        reduction: str,
        base_model: str = '', 
        dist_measure: str = '', 
        train_norm: str = '',
        subset: str = 'test'):
    
    return count_cases_of_increased_accuracy(
            pair_df = pair_df,
            reduction_from = 'base',
            reduction_to = reduction,
            base_model = base_model, 
            dist_measure = dist_measure, 
            train_norm = train_norm,
            subset = subset)


def make_paired_plot(
        pair_df: pd.DataFrame,
        base_model: str, 
        dist_measure: str, 
        train_norm: str,
        subset: str = 'test') -> None:
    
    reductions = get_hubness_reductions()
    no_base_reductions = [red for red in reductions if red != 'base']
    
    seeds_df = pair_df.loc[base_model, dist_measure, train_norm]
    
    for red in no_base_reductions:
        base_red_cols = [f'acc base {subset}', f'acc {red} {subset}']
        base_accs = seeds_df[f'acc base {subset}']
        red_accs = seeds_df[f'acc {red} {subset}']
        
        base_red_x = [base_red_cols[0] 
                      for i in range(len(base_accs))] + [base_red_cols[1] 
                                    for i in range(len(red_accs))]
        
        X_coords = np.array([0, 1])
        Y_coords = np.array([base_accs, red_accs])
        
        plt.figure(figsize=(7,7))
        plt.plot(X_coords, 
                 Y_coords, 
                 color='gray',
                 zorder=1)        
        colours = {base_red_cols[0]:'green', base_red_cols[1]:'orange'}
        plt.scatter(base_red_x, 
            np.concatenate((base_accs.values, red_accs.values)).astype(float),
            s=150,
            c=list(map(colours.get, base_red_x)),
            zorder = 2)
        plt.title(f'{base_model} {dist_measure} {train_norm} base accuracy vs {red}')


def make_paired_plots(
        output_folder: str, 
        dataset: str, 
        prefix: str = '',
        subsets: List[str] = ['test'],
        train_normalisations: List[str] = ['n']):
    
    model_names = naming.get_model_names_from_knn_results(output_folder, dataset)
    
    pair_df, _ = get_data_for_paired_comparison(
        output_folder = output_folder,
        dataset = dataset,
        prefix = prefix,
        subsets = subsets)
    
    all_base_models = ['microsoft-MiniLM-L12-H384-uncased','distilroberta-base', 'microsoft-mpnet-base']
    relevant_base_models = [base_model 
                            for base_model in all_base_models
                            if any([base_model in model_name 
                                   for model_name in model_names])]
    
    dist_measures =  ['cos', 'cos_dist', 'euclidean']    
    
    for base_model in relevant_base_models:
        for dist_measure in dist_measures:
            for train_norm in train_normalisations:
                for subset in subsets:
                    make_paired_plot(
                        pair_df, base_model, dist_measure, train_norm, subset)


def horizontally_stacked_paired_plots(
        pair_df: pd.DataFrame,
        dataset: str,
        base_model: str, 
        dist_measure: str, 
        train_norm: str,
        non_base_reductions: List[str] = ['normal_all', 'mutual_proximity', 'normal_all_mutual_proximity'],
        title: str = 'base vs reductions for each seed',
        subset: str = 'test',
        width: float = 15,
        height: float = 7.5) -> None:
        
    scale = height/7.5
    
    seeds_df = pair_df.loc[base_model, dist_measure, train_norm]
    
    x_values = []
    y_values = []
    x_coords = []
    y_coords = []
    
    colours = { 1:'#4daf4a', 2: '#ff7f00'}
    
    for red in non_base_reductions:
        base_error = 1 - seeds_df[f'acc base {subset}']
        red_error = 1 - seeds_df[f'acc {red} {subset}']
        
        base_red_x = [ 1 for i in range(len(base_error))] + [2 for i in range(len(red_error))]
        x_values.append(base_red_x)
        
        y_value = np.concatenate((base_error.values, red_error.values)).astype(float)
        y_values.append(y_value)        
        
        x_coord = np.array([1, 2])
        x_coords.append(x_coord)
        
        y_coord = np.array([base_error, red_error])
        y_coords.append(y_coord)
    
    min_y = min([min(val) for val in y_values])
    max_y = max([max(val) for val in y_values])
    
    if (max_y - min_y) < 0.035:
        y_range = [min_y - 0.1/100, max_y + 0.1/100]
        step_size = 0.005
        tick_start = np.around(y_range[0], 3)
    else:
        y_range = [min_y - 0.5/100, max_y + 0.5/100]
        step_size = 0.01
        tick_start = np.around(y_range[0], 2)        
    
    
    y_ticks = np.arange(tick_start, y_range[1], step_size)
    
    fig, axes = plt.subplots(
        1, 
        ncols = len(non_base_reductions), 
        figsize=(width, height),
        sharey = True)
    fig.suptitle(title,
                 fontsize = 34 * scale)    
    
    for i in range(len(non_base_reductions)):
        
        axes[i].set_ylim(y_range)
        axes[i].set_yticks(y_ticks)
        
        axes[i].plot(
            x_coords[i], 
            y_coords[i], 
            color='gray',
            zorder=1)
        
        axes[i].scatter(
            x_values[i], 
            y_values[i],
            s=150,
            c=list(map(colours.get, x_values[i])),
            zorder = 2)
        
        sub_title = get_reduction_short_name(non_base_reductions[i])
        if i == (len(non_base_reductions)-1)/2:
            sub_title = f'{base_model}_{dist_measure}_{train_norm}\n' + sub_title
        
        axes[i].set_title(
            sub_title, 
            fontsize = 28 * scale)
        axes[i].set_xticks([0.5, 1, 2, 2.5])
        axes[i].grid()
    
        axes[i].set_xticklabels(['', 'base', 'reduction', ''], fontsize = 20 * scale)
        axes[i].tick_params(axis='both', which='both', labelsize = 20 * scale)
    
    axes[0].set_ylabel('Error rate', fontsize = 28 * scale)
    fig.tight_layout()
    
    return fig, axes


def make_distance_measure_plot(
        pair_df: pd.DataFrame,
        dataset: str, 
        base_model: str,
        reduction: str,        
        dist_measures: List[str], 
        train_normalisations: List[str],
        subset: str = 'test',
        save_folder: str = ''):
    
    width = 8
    height = 7.5
    sup_title = f'Error rate without vs with normalisation \n{base_model}'
    file_name_prefix = f'{dataset}_{base_model}_{"_".join(dist_measures)}_{"_".join(train_normalisations).replace(",", "-")}_{reduction}_acc'
    
    x_values = []
    y_values = []
    colours = sns.color_palette('colorblind')
    colours_dict = {dist_measures[i]:colours[i] 
                    for i in range(len(dist_measures))}
    given_colour = []
    
    for train_norm in train_normalisations:
        for dist_measure in dist_measures:
            seeds_df = pair_df.loc[base_model, dist_measure, train_norm]
            err_values = 1 - seeds_df[f'acc {reduction} {subset}']
            y_values.extend(err_values.values)
            x_values.extend([f'{dist_measure} {train_norm}'
                             for i in range(len(err_values))])
            given_colour.extend([f'{dist_measure}' 
                                 for i in range(len(err_values))])
            
    x_values = np.array(x_values, dtype = object)
    y_values = np.array(y_values)
    
    # Checking spread
    mean_y = np.mean(y_values)
    min_y = np.min(y_values)
    max_y = np.max(y_values)
    
    mean_max_diff = max_y - mean_y
    mean_min_diff = mean_y - min_y
    
    if mean_min_diff > 3 * mean_max_diff or mean_max_diff > 3 * mean_min_diff:
        logging.warning('Large spread found plotting with and without deviating values')
        smallest_spread = np.min([mean_max_diff, mean_min_diff])
        spread_filter = np.absolute(y_values - mean_y) < 3*smallest_spread
        
        plt.figure(figsize = (width, height))
        plt.scatter(x_values[spread_filter], 
            y_values[spread_filter],
            s=150,
            c=list(map(colours_dict.get, np.array(given_colour)[spread_filter])))
        plt.suptitle(sup_title, fontsize = 34)
        plt.title(' \nnone: No normalisation, n: normalised to unit length, deviant values removed')
        
        if save_folder != '':
            file_name = f'{file_name_prefix}_deviants_removed'
            save_to_file.save_pdf_fig(file_name, save_folder)
    
    
    plt.figure(figsize = (width, height))
    plt.scatter(x_values, 
        y_values,
        s=150,
        c=list(map(colours_dict.get, given_colour)))
    plt.suptitle(sup_title, fontsize = 34)
    plt.title(' \nnone: No normalisation, n: normalised to unit length')
    if save_folder != '':
        file_name = f'{file_name_prefix}'
        save_to_file.save_pdf_fig(file_name, save_folder)


def make_distance_measure_reductions_plot(
        pair_df: pd.DataFrame,
        dataset: str, 
        base_model: str,
        reductions: Tuple[str, str],        
        dist_measures: List[str], 
        train_normalisations: List[str],
        subset: str = 'test',
        save_folder: str = ''):
    
    width = 15
    height = 7.5
    scale = height/7.5
    sup_title = f'Error rates without vs with normalisation \n{base_model}'
    file_name_prefix = f'{dataset}_{base_model}_{"_".join(dist_measures)}_{"_".join(train_normalisations).replace(",", "-")}_{reductions[0]}_{reductions[1]}_acc'
    
    
    fig, axes = plt.subplots(
        1, 
        ncols = 2, 
        figsize=(width, height),
        sharey = True)
    fig.suptitle(sup_title,
                 fontsize = 34 * scale)  
    
    x_tick_labels = [n for n in train_normalisations for i in dist_measures]
    colours = sns.color_palette('colorblind')
    colours_dict = {dist_measures[i]:colours[i] 
                    for i in range(len(dist_measures))}
    
    for i, reduction in enumerate(reductions):
        x_values = []
        y_values = []
        given_colour = []
        for train_norm in train_normalisations:
            for dist_measure in dist_measures:
                seeds_df = pair_df.loc[base_model, dist_measure, train_norm]
                err_values = 1 - seeds_df[f'acc {reduction} {subset}']
                y_values.extend(err_values.values)
                x_values.extend([f'{dist_measure} {train_norm}'
                                 for i in range(len(err_values))])
                given_colour.extend([f'{dist_measure}' 
                                     for i in range(len(err_values))])
                
        x_values = np.array(x_values, dtype = object)
        y_values = np.array(y_values)
        given_colour = np.array(given_colour)
    
        # Checking spread
        mean_y = np.mean(y_values)
        min_y = np.min(y_values)
        max_y = np.max(y_values)
        
        mean_max_diff = max_y - mean_y
        mean_min_diff = mean_y - min_y
        
        if mean_min_diff > 3 * mean_max_diff or mean_max_diff > 3 * mean_min_diff:
            logging.warning('Large spread found, will plot without deviating values')
            smallest_spread = np.min([mean_max_diff, mean_min_diff])
            spread_filter = np.absolute(y_values - mean_y) < 3*smallest_spread
            
            x_values = x_values[spread_filter]
            y_values = y_values[spread_filter]
            given_colour = np.array(given_colour)[spread_filter]
            
            file_name_prefix = file_name_prefix + '_deviants_removed'
        
        for train_norm in train_normalisations:
            for dist_measure in dist_measures:
                filt = x_values == f'{dist_measure} {train_norm}'
                label = dist_measure if train_norm == 'none' else ''
                axes[i].scatter(x_values[filt], 
                y_values[filt],
                s=150,
                c = colours_dict.get(given_colour[filt][0]),
                label = label)
                
                
        axes[i].grid(True)
        axes[i].tick_params(axis='both', which='both', labelsize = 20 * scale)
        axes[i].set_xticklabels(x_tick_labels)
        axes[i].set_title(get_reduction_short_name(reduction), fontsize = 28 * scale)
        
    axes[1].legend(loc = 'lower right', fontsize = 18 * scale)
    fig.text(0.51, -0.05, 'none: No normalisation, n: normalised to unit length', 
             ha='center',
             fontsize = 28 * scale)
    fig.tight_layout()
    if save_folder != '':
        file_name = f'{file_name_prefix}'
        save_to_file.save_pdf_fig(file_name, save_folder)


    
def make_distance_measure_plots_none_vs_norm(
        output_folder: str, 
        dataset: str,
        reductions: List[str],
        prefix: str = '',
        subsets: List[str] = ['test'],
        results_folder: str = '',
        save_folder: str = 'output/figures') -> None:
    
    model_names = naming.get_model_names_from_knn_results(
        output_folder, 
        dataset,
        prefix = prefix,
        results_folder = results_folder)
    
    pair_df, _ = get_data_for_paired_comparison(
        output_folder = output_folder,
        dataset = dataset,
        prefix = prefix,
        subsets = subsets,
        results_folder = results_folder)
    
    all_base_models = ['microsoft-MiniLM-L12-H384-uncased','distilroberta-base', 'microsoft-mpnet-base']
    relevant_base_models = [base_model 
                            for base_model in all_base_models
                            if any([base_model in model_name 
                                   for model_name in model_names])]
    
    dist_measures =  ['euclidean', 'cos', 'cos_dist']
    train_normalisations = ['none', 'n']
    
    for base_model in relevant_base_models:
        for reduction in reductions:
            make_distance_measure_plot(
                pair_df, dataset, base_model, reduction,
                dist_measures, train_normalisations, 
                save_folder = save_folder)
            

def make_distance_measure_plots_none_vs_norm_base_vs_comb(
        output_folder: str, 
        dataset: str,
        reductions: Tuple[str, str],
        prefix: str = '',
        subsets: List[str] = ['test'],
        results_folder: str = '',
        save_folder: str = 'output/figures') -> None:
    
    model_names = naming.get_model_names_from_knn_results(
        output_folder, 
        dataset,
        prefix = prefix,
        results_folder = results_folder)
    
    pair_df, _ = get_data_for_paired_comparison(
        output_folder = output_folder,
        dataset = dataset,
        prefix = prefix,
        subsets = subsets,
        results_folder = results_folder)
    
    all_base_models = ['microsoft-MiniLM-L12-H384-uncased','distilroberta-base', 'microsoft-mpnet-base']
    relevant_base_models = [base_model 
                            for base_model in all_base_models
                            if any([base_model in model_name 
                                   for model_name in model_names])]
    
    dist_measures =  ['euclidean', 'cos', 'cos_dist']
    train_normalisations = ['none', 'n']
    
    for base_model in relevant_base_models:
        make_distance_measure_reductions_plot(
            pair_df, dataset, base_model, reductions,
            dist_measures, train_normalisations, 
            save_folder = save_folder)            
    

def get_data_for_paired_comparison_pretrained_models(
        results_folder_path: str, 
        dataset: str,
        reductions: List['str'] = [],
        prefix: str = '',
        subsets: List[str] = ['test']) -> Tuple[pd.DataFrame, Dict[str, NDArray]]:
    
    model_names = naming.get_model_names_from_knn_results_and_results_path(
        results_folder_path, dataset)
    
    files_in_results = [file 
                        for file in os.listdir(results_folder_path)
                        if file.endswith('.json')]
    
    if not reductions:
        reductions = get_hubness_reductions_from_knn_results_results_path(
            results_folder_path, dataset, prefix)
    
    pair_correct_idx_dict = {}
    pair_df = pd.DataFrame(index = model_names)
    
    d_fold = 10
    
    for model_name in tqdm(model_names):
        
        hub_files = [file 
                     for file in os.listdir(results_folder_path)
                     if re.match(f'{model_name}_{dataset}_hubness_results.json', file)]
        if len(hub_files) != 1:
            raise ValueError(f'Found: {len(hub_files)} with model_name: {model_name} and dataset: {dataset}')
        
        hub_file = hub_files[0]
        
        hub_file_path = os.path.join(results_folder_path, hub_file)
        hub_json_dict = save_to_file.load_json(hub_file_path)
        
        
        for red in reductions: 
            force_dist, skhubness_reduction = convert_hubness_reduction_name(red)
            
            suffix = saving_results.get_result_name_suffix(
                False,
                skhubness_reduction,
                force_dist, 
                d_fold)
            
            model_result_files = [
                file 
                for file in files_in_results
                if re.match(f'{model_name}_{dataset}_knn_results{suffix}.json', file)]
            
            if len(model_result_files) != 1:
                raise ValueError(f'Found: {len(model_result_files)} with model_name: {model_name}, dataset: {dataset} and suffix: {suffix}')
            
            model_result_file = model_result_files[0]
            
            test_results = saving_results.get_knn_test_results_results_path(
                results_folder_path, model_result_file)
            test_results = saving_results.get_test_results_with_standard_keys(
                test_results)
            
            pair_correct_idx_dict[f'{model_name} {red}'] = test_results['all'].correct_test_idx 
            
            for subset in subsets:
                acc = test_results['all'].train_score if subset == 'train' else test_results['all'].test_score 
                a_col = f'acc {red} {subset}'
                pair_df.loc[model_name, a_col] = acc
                
                s_col = f'skew {red} {subset}'
                pair_df.loc[model_name, s_col] = hub_json_dict[red][subset]['k_skewness']
                
                r_col = f'rh {red} {subset}'
                pair_df.loc[model_name, r_col] = hub_json_dict[red][subset]['robinhood']
    
    return pair_df, pair_correct_idx_dict


def get_data_for_pretrained_models_table(
        results_folder_path: str, 
        dataset: str,
        reductions: List['str'] = [],
        prefix: str = '',
        subsets: List[str] = ['test']) -> pd.DataFrame:
    
    model_names = naming.get_model_names_from_knn_results_and_results_path(
        results_folder_path, dataset)
    
    files_in_results = [file 
                        for file in os.listdir(results_folder_path)
                        if file.endswith('.json')]
    
    if not reductions:
        reductions = get_hubness_reductions_from_knn_results_results_path(
            results_folder_path, dataset, prefix)
    
    index = []
    for name in model_names:
        for red in reductions:
            index.append(f'{name} {red}')
    
    
    pair_df = pd.DataFrame(index = index)
    
    d_fold = 10
    
    for model_name in tqdm(model_names):
        
        hub_files = [file 
                     for file in os.listdir(results_folder_path)
                     if re.match(f'{model_name}_{dataset}_hubness_results.json', file)]
        if len(hub_files) != 1:
            raise ValueError(f'Found: {len(hub_files)} with model_name: {model_name} and dataset: {dataset}')
        
        hub_file = hub_files[0]
        
        hub_file_path = os.path.join(results_folder_path, hub_file)
        hub_json_dict = save_to_file.load_json(hub_file_path)
        
        
        for red in reductions: 
            force_dist, skhubness_reduction = convert_hubness_reduction_name(red)
            
            suffix = saving_results.get_result_name_suffix(
                False,
                skhubness_reduction,
                force_dist, 
                d_fold)
            
            model_result_files = [
                file 
                for file in files_in_results
                if re.match(f'{model_name}_{dataset}_knn_results{suffix}.json', file)]
            
            if len(model_result_files) != 1:
                raise ValueError(f'Found: {len(model_result_files)} with model_name: {model_name}, dataset: {dataset} and suffix: {suffix}')
            
            model_result_file = model_result_files[0]
            
            test_results = saving_results.get_knn_test_results_results_path(
                results_folder_path, model_result_file)
            test_results = saving_results.get_test_results_with_standard_keys(
                test_results)
            
            for subset in subsets:
                acc = test_results['all'].train_score if subset == 'train' else test_results['all'].test_score 
                a_col = 'accuracy'
                pair_df.loc[f'{model_name} {red}', a_col] = acc
                
                c_col = 'confidence'
                plus_minus = get_confidence_interval(acc, dataset, '95%')
                pair_df.loc[f'{model_name} {red}', c_col] = plus_minus
                
                s_col = 'k-skewness'
                pair_df.loc[f'{model_name} {red}', s_col] = hub_json_dict[red][subset]['k_skewness']
                
                r_col = 'robinhood score'
                pair_df.loc[f'{model_name} {red}', r_col] = hub_json_dict[red][subset]['robinhood']
                
                
    
    return pair_df


def get_confidence_interval(accuracy: float, dataset: str, significance: str = '95%'):
    if dataset == 'newsgroups':
        n = 7532
    elif dataset == 'ag_news':
        n = 7600
    elif dataset == 'yahoo_answers_small':
        n = 6000
    elif dataset == 'yahoo_answers':
        n = 60000
    else:
        raise ValueError(f'Dataset not recognized: {dataset}')
    
    if significance == '90%':
        z = 1.64
    elif significance == '95%':
        z = 1.96
    elif significance == '98%':
        z = 2.33
    elif significance == '99%':
        z = 2.58
    else:
        raise ValueError(f'Significance not recognized: {significance}')
    
    plus_minus = z * math.sqrt(accuracy * (1-accuracy)/ n)
    return plus_minus
    

def make_hubness_vs_acc_plot(
        pair_df: pd.DataFrame,
        hubness_measure: str,
        base_model: str,
        reduction: str,
        dist_measure: str = '', 
        train_norm: str = '',
        subset_hubness: str = 'test',
        include_base: bool = True):
    
    if base_model != '':
        pair_df = pair_df.xs(base_model, drop_level = False)
    if dist_measure != '':
        pair_df = pair_df.xs(dist_measure, level = 1, drop_level = False)
    if train_norm != '':
        pair_df = pair_df.xs(train_norm, level = 2, drop_level = False)
    
    if include_base:
        acc_values = np.concatenate((pair_df['acc base test'].values, pair_df[f'acc {reduction} test'].values))
        hub_values = np.concatenate((pair_df[f'{hubness_measure} base {subset_hubness}'].values, pair_df[f'{hubness_measure} {reduction} {subset_hubness}'].values))
    else:
        acc_values = pair_df[f'acc {reduction} test'].values
        hub_values = pair_df[f'{hubness_measure} {reduction} test'].values
    
    plt.figure(figsize=(7,7))
    plt.scatter(hub_values, 
        acc_values,
        s=150)
    plt.xlabel('accuracy')
    plt.ylabel(hubness_measure)
    
    plt.title(f'{base_model} {dist_measure} {train_norm} accuracy vs {hubness_measure}')
    
    plt.show()


def make_delta_hubness_vs_delta_acc_plot(
        pair_df: pd.DataFrame,
        hubness_measure: str,
        base_model: str,
        reduction: str,
        dist_measure: str = '', 
        train_norm: str = '',
        subset_hubness: str = 'test'):
    
    if base_model != '':
        pair_df = pair_df.xs(base_model, drop_level = False)
    if dist_measure != '':
        pair_df = pair_df.xs(dist_measure, level = 1, drop_level = False)
    if train_norm != '':
        pair_df = pair_df.xs(train_norm, level = 2, drop_level = False)
    
    delta_acc = pair_df[f'acc {reduction} test'].values - pair_df['acc base test'].values
    delta_hub = pair_df[f'{hubness_measure} base {subset_hubness}'].values - pair_df[f'{hubness_measure} {reduction} {subset_hubness}'].values
    
    plt.figure(figsize=(7,7))
    plt.scatter(delta_hub, 
        delta_acc,
        s=150)
    plt.xlabel(f'minus delta {hubness_measure}' )
    plt.ylabel('delta accuracy')
    
    plt.title(f'{base_model} {dist_measure} {train_norm} delta accuracy vs minus delta {hubness_measure}')
    
    plt.show()
    


def make_delta_hubness_vs_delta_acc_plot_percent(
        pair_df: pd.DataFrame,
        hubness_measure: str,
        base_model: str,
        reduction: str,
        dist_measure: str = '', 
        train_norm: str = '',
        subset_hubness: str = 'test'):
    
    if base_model != '':
        pair_df = pair_df.xs(base_model, drop_level = False)
    if dist_measure != '':
        pair_df = pair_df.xs(dist_measure, level = 1, drop_level = False)
    if train_norm != '':
        pair_df = pair_df.xs(train_norm, level = 2, drop_level = False)
    
    base_acc = pair_df['acc base test'].values
    base_hub = pair_df[f'{hubness_measure} base {subset_hubness}'].values
    
    delta_acc = pair_df[f'acc {reduction} test'].values - base_acc
    delta_hub = base_hub - pair_df[f'{hubness_measure} {reduction} {subset_hubness}'].values
    
    # delta_acc_percent = (delta_acc/base_acc) * 100
    delta_hub_percent = (delta_hub/base_hub) * 100
    
    plt.figure(figsize=(10,10))
    plt.scatter(delta_hub_percent, 
        delta_acc,
        s=150)
    plt.xlabel(f'minus delta {hubness_measure} percent' )
    plt.ylabel('delta accuracy')
    
    s_coef, s_p_val = spearmanr(delta_hub_percent, delta_acc)
    p_coef, p_p_val = pearsonr(delta_hub_percent, delta_acc)
    
    s_p_val_text = f'p-value: {format(s_p_val, ".4f")}' if s_p_val >= 0.0001 else 'p-value < 0.0001'
    p_p_val_text = f'p-value: {format(p_p_val, ".4f")}' if p_p_val >= 0.0001 else 'p-value < 0.0001'
        
    plt.suptitle(
        f'{reduction} \n {base_model} {dist_measure} {train_norm} delta accuracy vs minus delta {hubness_measure} percent',
        fontsize = 34)
    plt.title(
        f'spearman coef: {format(s_coef, ".4f")}, {s_p_val_text}, pearson coef: {format(p_coef, ".4f")}, {p_p_val_text}',
        fontsize = 14)
    
    plt.show()
    
    """
    plt.figure(figsize=(10,10))
    plt.scatter(delta_hub_percent, 
        delta_acc_percent,
        s=150)
    plt.xlabel(f'minus delta {hubness_measure} percent' )
    plt.ylabel('delta accuracy percent')
    
    s_coef, s_p_val = spearmanr(delta_hub_percent, delta_acc_percent)
    p_coef, p_p_val = pearsonr(delta_hub_percent, delta_acc_percent)
    
    plt.suptitle(
        f'{reduction} {base_model} {dist_measure} {train_norm} delta accuracy vs minus delta {hubness_measure} percent',
        fontsize = 20)
    plt.title(
        f'spearman coef: {format(s_coef, ".4f")}, p-value: {format(s_p_val, ".5f")}, pearson coef: {format(p_coef, ".4f")}, p-value: {format(p_p_val, ".5f")}',
        fontsize = 14)
    
    plt.show()   
    """


def get_results_with_features(
        pair_df: pd.DataFrame,
        base_models: List[str],
        dist_measures: List[str], 
        train_norms: List[str]):
    
    if base_models:
        pair_df = pair_df.loc[base_models, :, :, :]
    
    if dist_measures:
        pair_df = pair_df.loc[:, dist_measures, :, :]
        
    if train_norms:
        pair_df = pair_df.loc[:, :, train_norms, :]
        
    return pair_df



def make_k_occurrence_dist_plot(
        result_folder_path: str,
        dataset: str,
        base_model: str,
        seed: int,
        reductions: List[str],
        dist_measure: str = 'euclidean', 
        train_norm: str = 'n',
        subset: str = 'train',
        width = 20,
        height = 5,
        save_folder: str = 'output/figures/',
        file_name_suffix: str = '',
        plot_vs_std_norm: bool = False):
    
    model_name = naming.get_sts_bert_model_save_name(
        base_model,
        dist_measure,
        vectors.VectorRelation.ORTHOGONAL.name,
        True if 'z' in train_norm else False,
        True if 'n' in train_norm else False,
        True if 'c' in train_norm else False,
        seed)
    
    #short_model_name = f'{base_model} {dist_measure} {train_norm} seed{seed}'
    
    results_name = saving_results.get_result_name(
        'hubness',
        model_name, 
        dataset,
        center_norm_after_train = False,
        skhubness_reduction = SkhubnessReduction.NONE,
        force_dist = ForceDistribution.NONE,
        d_fold = 0)
    
    result_path = os.path.join(result_folder_path, results_name)
    
    hub_json_dict = save_to_file.load_json(result_path)
    
    k_occurrences = [np.array(hub_json_dict[r]['train']['k_occurrence'])
                     for r in reductions]
    
    title_suffix = ' vs standard normal distribution normalised to unit length' if plot_vs_std_norm else ''
    
    fig, axes = analysis.horizontally_stacked_N_k_plots_from_N_k_array(
            N_k_array = k_occurrences,
            title = f'K-occurrence - embeddings{title_suffix}',
            sub_titles = [get_reduction_short_name(r) for r in reductions],
            log_scale_y = [True for r in reductions],
            log_scale_x = [True for r in reductions],
            always_log = True,
            width = width,
            height = height,
            plot_vs_std_norm = plot_vs_std_norm)
     
    save_prefix = f'{base_model}_{dist_measure}_{train_norm}_seed{seed}'
    file_name = f'{save_prefix}_norm_all_MP_and_comb_{dataset}_{file_name_suffix}'
    
    save_to_file.save_pdf_fig(file_name, save_folder)
        
    
    hubness_scores = {}
    for r in reductions:
        hubness_scores[r] = {}
        hubness_scores[r]['k_skewness'] = hub_json_dict[r]['train']['k_skewness']
        hubness_scores[r]['robinhood'] = hub_json_dict[r]['train']['robinhood']
        
    hubness_scores_path = os.path.join(save_folder, f'{save_prefix}_hubness_scores_{file_name_suffix}.json')
    save_to_file.save_as_json(hubness_scores, save_to = hubness_scores_path)
        
    return fig, axes


def get_top_models_with_acc_from_overview(
        overview_df: pd.DataFrame,
        top_n: int = 3,
        stat: str = 'mean'):
    acc_stat = f'test acc {stat}'
    accuracies = overview_df[acc_stat].values
    
    top_acc_idx = accuracies.argsort()[-top_n:]
    top_acc = accuracies[top_acc_idx]

    top_models = overview_df[overview_df[acc_stat] >= min(top_acc)]
    
    top_models_sorted = top_models.iloc[np.flip(top_models[acc_stat].argsort())]
        
    return top_models_sorted
    
       




