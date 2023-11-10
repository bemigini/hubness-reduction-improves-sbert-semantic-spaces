# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:30:39 2022

@author: bmgi


Making figures for article


"""


from dataclasses import dataclass

import matplotlib.pyplot as plt

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

import os
import pandas as pd

from scipy.stats import beta
from scipy.stats import binom
#import statsmodels.api as sm
from sklearn import preprocessing
from statsmodels.stats.contingency_tables import mcnemar

from src import sentence_bert as sent_bert
from src.hubness import analysis, reduction, synthetic_data, visualization
from src.performance import final_results
from src.util import save_to_file


from typing import List, Tuple



def error_and_hubness_reduction_plots():
    output_folder = 'output'
    prefix = ''
    subsets = ['test']
    save_folder = 'output/figures'    
    
    
    dataset = 'newsgroups'
    results_folder = 'r_20_st'

    pair_20_df, _ = final_results.get_data_for_paired_comparison(
        output_folder,
        dataset,
        reductions = [],
        prefix = prefix,
        subsets = subsets,
        results_folder = results_folder)


    dataset = 'ag_news'
    results_folder = 'r_ag'

    pair_ag_df, _ = final_results.get_data_for_paired_comparison(
        output_folder,
        dataset,
        reductions = [],
        prefix = prefix,
        subsets = subsets,
        results_folder = results_folder)



    dataset = 'yahoo_answers_small'
    results_folder = 'r_ya'

    pair_ya_df, _ = final_results.get_data_for_paired_comparison(
        output_folder,
        dataset,
        reductions = [],
        prefix = prefix,
        subsets = subsets,
        results_folder = results_folder)
    
    
    error_rate_reduction_plot_for_3_methods(
        ['normal_all', 'mutual_proximity', 'normal_all_mutual_proximity'],
        save_folder,
        pair_20_df,
        pair_ag_df,
        pair_ya_df)
    
    
    hubness_reduction_plot_for_3_methods(
        ['normal_all', 'mutual_proximity', 'normal_all_mutual_proximity'],
        ['skew', 'rh'],
        save_folder,
        pair_20_df,
        pair_ag_df,
        pair_ya_df)
    


def error_rate_reduction_plot_for_3_methods(
        method_names: List[str],
        save_folder: str,
        pair_20_df,
        pair_ag_df,
        pair_ya_df) -> None:
    
    method_to_title_name = {
        'normal_all': 'f-norm', 
        'mutual_proximity': 'MP',
        'normal_all_mutual_proximity': 'f-norm + MP'}
    
    fig, axes = plt.subplots(
        nrows = 1, 
        ncols = len(method_names),        
        figsize=(15, 7.5))
    
    
    for i in range(len(axes)):
        method = method_names[i]
        
        val_20, acc_diff_20, val_ag, acc_diff_ag, val_ya, acc_diff_ya = get_error_diffs(method, pair_20_df, pair_ag_df, pair_ya_df)
        
        skew_filter_20_0 = pair_20_df['skew base test'] <= 2.5
        skew_filter_20_1 = (3 >= pair_20_df['skew base test']) & (pair_20_df['skew base test'] > 2.5)
        skew_filter_20_2 = pair_20_df['skew base test'] > 3
                
        skew_filter_ag_0 = pair_ag_df['skew base test'] <= 2.5
        skew_filter_ag_1 = (3 >= pair_ag_df['skew base test']) & (pair_ag_df['skew base test'] > 2.5)
        skew_filter_ag_2 = pair_ag_df['skew base test'] > 3
        
        skew_filter_ya_0 = pair_ya_df['skew base test'] <= 2.5
        skew_filter_ya_1 = (3 >= pair_ya_df['skew base test']) & (pair_ya_df['skew base test'] > 2.5)
        skew_filter_ya_2 = pair_ya_df['skew base test'] > 3
        
        axes[i].axhline(y=0., color='k', linestyle='-')
            
        axes[i].plot(val_20[skew_filter_20_0], acc_diff_20.values[skew_filter_20_0], 'o', color='#984ea3', markersize=5, label='2.5 >= k-skew')
        axes[i].plot(val_20[skew_filter_20_1], acc_diff_20.values[skew_filter_20_1], 'o', color='#ff7f00', markersize=5, label='3 >= k-skew > 2.5')
        axes[i].plot(val_20[skew_filter_20_2], acc_diff_20.values[skew_filter_20_2], 'o', color='#4daf4a', markersize=5, label='k-skew > 3')
        
        axes[i].plot(val_ag[skew_filter_ag_0], acc_diff_ag.values[skew_filter_ag_0], 'o', color='#984ea3', markersize=5)
        axes[i].plot(val_ya[skew_filter_ya_0], acc_diff_ya.values[skew_filter_ya_0], 'o', color='#984ea3', markersize=5)
        
        axes[i].plot(val_ag[skew_filter_ag_2], acc_diff_ag.values[skew_filter_ag_2], 'o', color='#4daf4a', markersize=5)
        axes[i].plot(val_ya[skew_filter_ya_2], acc_diff_ya.values[skew_filter_ya_2], 'o', color='#4daf4a', markersize=5)
        
        axes[i].plot(val_ag[skew_filter_ag_1], acc_diff_ag.values[skew_filter_ag_1], 'o', color='#ff7f00', markersize=5)
        axes[i].plot(val_ya[skew_filter_ya_1], acc_diff_ya.values[skew_filter_ya_1], 'o', color='#ff7f00', markersize=5)

        
        if i > 0:
            axes[i].set_yticklabels([])
        else:
            axes[i].set_ylabel('Error rate reduction \n in percent', fontsize = 24)
            
        
        axes[i].set_xticks([0, 1, 2])
        axes[i].set_xticklabels(['20', 'AG', 'YA'], fontsize = 20)
        axes[i].tick_params(axis='both', which='both', labelsize = 20)
        axes[i].grid()
        
        axes[i].set_title(f'{method_to_title_name[method]}', fontsize = 24)
        
        axes[i].set_ylim(-11, 27)
        
        if i == 1:            
            legend = axes[i].legend(fontsize = 18)
    
    plt.subplots_adjust(wspace=0.05, hspace=0)
            
    
    save_to_file.save_high_res_png('Error_rate_reduction_from_using_f-norm_MP_both', save_folder)
    
        
    


def error_rate_reduction_plot_for_method(
        method_name: str,  save_name: str,
        save_folder: str,
        pair_20_df,
        pair_ag_df,
        pair_ya_df) -> None:
    
    
    acc_diff_20 = (pair_20_df[f'acc {method_name} test'] - pair_20_df['acc base test'])*100 /(1-pair_20_df['acc base test'])
    acc_diff_ag = (pair_ag_df[f'acc {method_name} test'] - pair_ag_df['acc base test'])*100 /(1-pair_ag_df['acc base test'])
    acc_diff_ya = (pair_ya_df[f'acc {method_name} test'] - pair_ya_df['acc base test'])*100 /(1-pair_ya_df['acc base test'])
    
    rng = np.random.default_rng(42)
    
    size_20 = acc_diff_20.shape[0]      
    noise_20 = rng.uniform(-0.3, 0.3, size_20)
    val_20 = np.zeros(size_20) + noise_20
    skew_filter_20_0 = pair_20_df['skew base test'] <= 2.5
    skew_filter_20_1 = (3 >= pair_20_df['skew base test']) & (pair_20_df['skew base test'] > 2.5)
    skew_filter_20_2 = pair_20_df['skew base test'] > 3
    
    size_ag = acc_diff_ag.shape[0]      
    noise_ag = rng.uniform(-0.3, 0.3, size_ag)
    val_ag = np.ones(size_ag) + noise_ag
    skew_filter_ag_0 = pair_ag_df['skew base test'] <= 2.5
    skew_filter_ag_1 = (3 >= pair_ag_df['skew base test']) & (pair_ag_df['skew base test'] > 2.5)
    skew_filter_ag_2 = pair_ag_df['skew base test'] > 3
    
    noise_ya = rng.uniform(-0.3, 0.3, size_ag)
    val_ya = np.ones(size_ag) + 1 + noise_ya
    skew_filter_ya_0 = pair_ya_df['skew base test'] <= 2.5
    skew_filter_ya_1 = (3 >= pair_ya_df['skew base test']) & (pair_ya_df['skew base test'] > 2.5)
    skew_filter_ya_2 = pair_ya_df['skew base test'] > 3
    
    fig, axes = plt.subplots(
        1, 
        ncols = 1, 
        figsize=(15, 7.5))
    fig.suptitle(f'Error rate reduction from using {save_name.replace("_", " + ")}',
                 fontsize = 30)    
    
    axes.axhline(y=0., color='k', linestyle='-')
        
    axes.plot(val_20[skew_filter_20_0], acc_diff_20.values[skew_filter_20_0], 'o', color='#ff7f00', markersize=6, label='2.5 >= k-skew')
    axes.plot(val_ag[skew_filter_ag_0], acc_diff_ag.values[skew_filter_ag_0], 'o', color='#ff7f00', markersize=6)
    axes.plot(val_ya[skew_filter_ya_0], acc_diff_ya.values[skew_filter_ya_0], 'o', color='#ff7f00', markersize=6)
    
    axes.plot(val_20[skew_filter_20_1], acc_diff_20.values[skew_filter_20_1], 'o', color='#4daf4a', markersize=6, label='3 >= k-skew > 2.5')
    axes.plot(val_ag[skew_filter_ag_1], acc_diff_ag.values[skew_filter_ag_1], 'o', color='#4daf4a', markersize=6)
    axes.plot(val_ya[skew_filter_ya_1], acc_diff_ya.values[skew_filter_ya_1], 'o', color='#4daf4a', markersize=6)
        
    axes.plot(val_20[skew_filter_20_2], acc_diff_20.values[skew_filter_20_2], 'o', color='#377eb8', markersize=6, label='k-skew > 3')
    axes.plot(val_ag[skew_filter_ag_2], acc_diff_ag.values[skew_filter_ag_2], 'o', color='#377eb8', markersize=6)
    axes.plot(val_ya[skew_filter_ya_2], acc_diff_ya.values[skew_filter_ya_2], 'o', color='#377eb8', markersize=6)
        
    
    axes.set_xticks([0, 1, 2])
    axes.set_xticklabels(['20', 'AG', 'YA'], fontsize = 20)
    axes.tick_params(axis='both', which='both', labelsize = 20)
    
    axes.set_ylim(-11, 27)
    
    axes.set_ylabel('Error rate reduction \n in percent', fontsize = 24)
    axes.legend()
    
    legend = axes.legend(fontsize = 18)
    plt.setp(legend.get_title(), fontsize = 18)
    
    
    save_to_file.save_high_res_png(f'Error_rate_reduction_from_using_{save_name}', save_folder)


def hubness_reduction_plot_for_3_methods(
        method_names: List[str],
        hub_scores: List[str],
        save_folder: str,
        pair_20_df,
        pair_ag_df,
        pair_ya_df) -> None:
    
    method_to_title_name = {
        'normal_all': 'f-norm', 
        'mutual_proximity': 'MP',
        'normal_all_mutual_proximity': 'f-norm + MP'}
    hub_score_to_label_name = {
        'skew': 'k-skew',
        'rh': 'rh'}
    
    fig, axes = plt.subplots(
        nrows = len(hub_scores), 
        ncols = len(method_names),        
        figsize=(15, 7.5))
    
    
    for i in range(len(axes)):
        for j in range(len(axes[0])):
            method = method_names[j]        
            hub_score = hub_scores[i]
            
            val_20, hub_diff_20, val_ag, hub_diff_ag, val_ya, hub_diff_ya = get_hub_diffs(method, hub_score, pair_20_df, pair_ag_df, pair_ya_df)
            
            axes[i, j].axhline(y=0., color='k', linestyle='-')
                            
            axes[i, j].plot(val_20, hub_diff_20.values, 'o', color='#377eb8', markersize=5) #, label='k-skew > 3')
            axes[i, j].plot(val_ag, hub_diff_ag.values, 'o', color='#377eb8', markersize=5)
            axes[i, j].plot(val_ya, hub_diff_ya.values, 'o', color='#377eb8', markersize=5)
            
            axes[i, j].set_xticks([0, 1, 2])
            if i == 1:                
                axes[i, j].set_xticklabels(['20', 'AG', 'YA'], fontsize = 20)
            else:
                axes[i, j].set_xticklabels([], fontsize = 20)
            
            if j > 0:
                axes[i, j].set_yticklabels([])
            else:
                axes[i, j].set_ylabel(f'{hub_score_to_label_name[hub_score]} reduction \n in percent', fontsize = 24)
                axes[i, j].yaxis.set_label_coords(-0.25,0.55)
            
            axes[i, j].tick_params(axis='both', which='both', labelsize = 20)
            axes[i, j].grid()
            
            if hub_score == 'skew':
                axes[i, j].set_ylim(-150, 100)
                axes[i, j].set_yticks([-100, -50, 0, 50, 100])
            else:
                axes[i, j].set_ylim(-10, 50)
            
            if i == 0:
                axes[i, j].set_title(f'{method_to_title_name[method]}', fontsize = 24)
            
    
    fig.suptitle('Hubness reduction in percent', fontsize = 30)
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    #fig.ylabel('Hubness reduction \n in percent', fontsize = 24)
    #fig.text(0.042, 0.30, 'rh', 
    #         va='center', 
    #         rotation='vertical',
    #         fontsize = 24)
      
    save_to_file.save_high_res_png('hubness_reduction_in_skew_rh_from_using_reductions', save_folder)
    
    

    
def get_hub_diffs(
        method_name: str,
        hub_score: str,
        pair_20_df,
        pair_ag_df,
        pair_ya_df):
    hub_diff_20 = (pair_20_df[f'{hub_score} base test'] - pair_20_df[f'{hub_score} {method_name} test'])*100 /pair_20_df[f'{hub_score} base test']
    hub_diff_ag = (pair_ag_df[f'{hub_score} base test'] - pair_ag_df[f'{hub_score} {method_name} test'])*100 /pair_ag_df[f'{hub_score} base test']
    hub_diff_ya = (pair_ya_df[f'{hub_score} base test'] - pair_ya_df[f'{hub_score} {method_name} test'])*100 /pair_ya_df[f'{hub_score} base test']
    
    rng = np.random.default_rng(42)
    
    size_20 = hub_diff_20.shape[0]      
    noise_20 = rng.uniform(-0.3, 0.3, size_20)
    val_20 = np.zeros(size_20) + noise_20
    
    
    size_ag = hub_diff_ag.shape[0]      
    noise_ag = rng.uniform(-0.3, 0.3, size_ag)
    val_ag = np.ones(size_ag) + noise_ag
    
    
    noise_ya = rng.uniform(-0.3, 0.3, size_ag)
    val_ya = np.ones(size_ag) + 1 + noise_ya
    
    return val_20, hub_diff_20, val_ag, hub_diff_ag, val_ya, hub_diff_ya


def get_error_diffs(
        method_name: str,
        pair_20_df,
        pair_ag_df,
        pair_ya_df):
    acc_diff_20 = (pair_20_df[f'acc {method_name} test'] - pair_20_df['acc base test'])*100 /(1-pair_20_df['acc base test'])
    acc_diff_ag = (pair_ag_df[f'acc {method_name} test'] - pair_ag_df['acc base test'])*100 /(1-pair_ag_df['acc base test'])
    acc_diff_ya = (pair_ya_df[f'acc {method_name} test'] - pair_ya_df['acc base test'])*100 /(1-pair_ya_df['acc base test'])
    
    rng = np.random.default_rng(42)
    
    size_20 = acc_diff_20.shape[0]      
    noise_20 = rng.uniform(-0.3, 0.3, size_20)
    val_20 = np.zeros(size_20) + noise_20
    
    
    size_ag = acc_diff_ag.shape[0]      
    noise_ag = rng.uniform(-0.3, 0.3, size_ag)
    val_ag = np.ones(size_ag) + noise_ag
    
    
    noise_ya = rng.uniform(-0.3, 0.3, size_ag)
    val_ya = np.ones(size_ag) + 1 + noise_ya    
    
    return val_20, acc_diff_20, val_ag, acc_diff_ag, val_ya, acc_diff_ya


def hubness_reduction_plot_for_method(
        method_name: str,  save_name: str,
        hub_score: str,
        save_folder: str,
        pair_20_df,
        pair_ag_df,
        pair_ya_df) -> None:
    
    
    hub_diff_20 = (pair_20_df[f'{hub_score} base test'] - pair_20_df[f'{hub_score} {method_name} test'])*100 /pair_20_df[f'{hub_score} base test']
    hub_diff_ag = (pair_ag_df[f'{hub_score} base test'] - pair_ag_df[f'{hub_score} {method_name} test'])*100 /pair_ag_df[f'{hub_score} base test']
    hub_diff_ya = (pair_ya_df[f'{hub_score} base test'] - pair_ya_df[f'{hub_score} {method_name} test'])*100 /pair_ya_df[f'{hub_score} base test']
    
    rng = np.random.default_rng(42)
    
    size_20 = hub_diff_20.shape[0]      
    noise_20 = rng.uniform(-0.3, 0.3, size_20)
    val_20 = np.zeros(size_20) + noise_20
    
    
    size_ag = hub_diff_ag.shape[0]      
    noise_ag = rng.uniform(-0.3, 0.3, size_ag)
    val_ag = np.ones(size_ag) + noise_ag
    
    
    noise_ya = rng.uniform(-0.3, 0.3, size_ag)
    val_ya = np.ones(size_ag) + 1 + noise_ya
    
    
    
    fig, axes = plt.subplots(
        1, 
        ncols = 1, 
        figsize=(15, 7.5))
    if hub_score == 'skew':
        hub_title = 'k-skewness'
    else:
        hub_title = 'robinhood score'
    
    fig.suptitle(f'Reduction in {hub_title} from using {save_name.replace("_", " + ")}',
                 fontsize = 30)    
    
    axes.axhline(y=0., color='k', linestyle='-')
        
        
    axes.plot(val_20, hub_diff_20.values, 'o', color='#377eb8', markersize=6) #, label='k-skew > 3')
    axes.plot(val_ag, hub_diff_ag.values, 'o', color='#377eb8', markersize=6)
    axes.plot(val_ya, hub_diff_ya.values, 'o', color='#377eb8', markersize=6)
       
    
    axes.set_xticks([0, 1, 2])
    axes.set_xticklabels(['20', 'AG', 'YA'], fontsize = 20)
    axes.tick_params(axis='both', which='both', labelsize = 20)
    
    if hub_score == 'skew':
        axes.set_ylim(-150, 100)
    else:
        axes.set_ylim(-10, 50)
    
    
    axes.set_ylabel('Hubness reduction \n in percent', fontsize = 24)
    
    save_to_file.save_high_res_png(f'hubness_reduction_in_{hub_score}_from_using_{save_name}', save_folder)



@dataclass
class McNemarResults():
    model_name: str 
    dataset: str
    comparison: Tuple[str, str]
    n_1_2: int
    n_2_1: int
    # Estimated difference in accuracy
    est_diff_accuracy: float
    p_value: float
    # alpha = 0.05 confidence interval
    conf_int: Tuple[float, float]


def mcnemar_test_pretrained():    
    #output_folder = 'output'
    prefix = ''
    subsets = ['test']
    #save_folder = 'output/figures'    
    
    
    dataset = 'newsgroups'
    results_folder_path = 'output/results_20_newsgroups_pretrained'
    

    pair_pre_20_df, correct_idx_dict = final_results.get_data_for_paired_comparison_pretrained_models(
        results_folder_path,
        dataset,
        reductions = [],
        prefix = prefix,
        subsets = subsets)
    
    
    model_names = list(set([k.split()[0] for k in correct_idx_dict]))
    comparisons = [('base', 'normal_all'), ('base', 'mutual_proximity'), ('base', 'normal_all_mutual_proximity')]
    
    if dataset == 'newsgroups':
        test_size = 7532
    elif dataset == 'ag_news':
        test_size = 7600
    elif dataset == 'yahoo_answers_small':
        test_size = 6000
    else: 
        raise ValueError(f'Dataset not recognized: {dataset}')
    
    mcnemar_res = []
    
    for model_name in model_names:
        for comp in comparisons:
            first_correct = correct_idx_dict[f'{model_name} {comp[0]}'][0]
            second_correct = correct_idx_dict[f'{model_name} {comp[1]}'][0]
            
            first_correct_second_wrong = [idx 
                                          for idx in first_correct
                                          if idx not in second_correct]
            second_correct_first_wrong = [idx 
                                          for idx in second_correct
                                          if idx not in first_correct]
            
            # McNemar's test
            n_1_2 = len(first_correct_second_wrong)
            n_2_1 = len(second_correct_first_wrong)
            if n_1_2 + n_2_1 < 5:
                raise ValueError(f'Too few differences: n_1_2: {n_1_2}, n_2_1: {n_2_1}')
            
            est_diff_accuracy = (n_1_2 - n_2_1)/test_size            
            p_value = 2*binom.cdf(min(n_1_2, n_2_1), n_1_2 + n_2_1, 0.5)
            
            # Alternative with scipy
            all_indexes = np.arange(test_size)
            both_correct = [idx 
                            for idx in all_indexes
                            if idx in first_correct and idx in second_correct]
            
            both_wrong = [idx 
                            for idx in all_indexes
                            if idx not in first_correct and idx not in second_correct]
            
            
            mc_table = [[len(both_correct), n_1_2],
                     [n_2_1, len(both_wrong)]]            
            mc_res = mcnemar(mc_table, exact=True)
            
            
            
            # Confidence interval
            alpha = 0.05
            Q = (test_size**2*(test_size + 1)*(est_diff_accuracy + 1)*(1 - est_diff_accuracy)) / (test_size*(n_1_2 + n_2_1) - (n_1_2 - n_2_1)**2)
            a = (est_diff_accuracy + 1) * (Q - 1) / 2
            b = (1 - est_diff_accuracy) * (Q - 1) / 2
            
            lower_bound = 2 * beta.ppf(alpha/2, a, b) - 1
            upper_bound = 2 * beta.ppf(1 - alpha/2, a, b) - 1
            
            result = McNemarResults(
                model_name, dataset, comp, 
                n_1_2, n_2_1, 
                est_diff_accuracy, 
                p_value,
                (lower_bound, upper_bound))
            
            mcnemar_res.append(result)
    
    save_to_file.save_as_json(mcnemar_res, 'output/mcnemar_newsgroups_pretrained.json')
            
    return mcnemar_res




def base_vs_reductions_plots_for_article_from_df(
        pair_df: pd.DataFrame,
        dataset: str,
        prefix: str = '',
        save_folder: str = 'output/figures') -> None:
    
    if dataset == 'newsgroups':
        base_models = ['microsoft-MiniLM-L12-H384-uncased','distilroberta-base', 'microsoft-mpnet-base']
    else:
        base_models = ['distilroberta-base', 'microsoft-mpnet-base']
    
    dist_measures = ['cos', 'cos_dist', 'euclidean']
    train_norms = ['none', 'c', 'n', 'c, n', 'z']
    
    for base_model in base_models:
        if prefix != '' and base_model not in set(pair_df.index.get_level_values(0)):
            continue
        for dist_measure in dist_measures:
            for train_norm in train_norms:
                final_results.horizontally_stacked_paired_plots(
                        pair_df = pair_df,
                        dataset = dataset,
                        base_model = base_model, 
                        dist_measure = dist_measure, 
                        train_norm = train_norm)
                
                file_name = f'{base_model}_{dist_measure}_{train_norm.replace(", ", "_")}_base_vs_reductions_{dataset}_article'
                save_to_file.save_pdf_fig(file_name, save_folder)


def synthetic_data_plots_for_article_abbreviated():
    random_state = 0
    rng = default_rng(random_state)
    
    save_folder = 'output/figures'
    
    k = 10
    dimensions = [3, 20, 768]    
    
    std_normal_name = 'std_normal_no_norm'
    std_normal_norm_name = 'std_normal_norm_emb'    
    
    k_skew_and_rh_dict = {}
    
    k_skew_and_rh_dict[std_normal_name] = {}
    k_skew_and_rh_dict[std_normal_norm_name] = {}
    
    std_N_k_results_no_norm = []
    std_N_k_results = []
    
    for dim in dimensions:
        std_normal, normal_mean_1, various_means = synthetic_data.make_normal_dist_data(dim, rng)
                
        N_k_result, hub_score_dict = analysis.get_k_occurrence_and_hubness_score_dict(
            std_normal, k)
        
        k_skew_and_rh_dict[std_normal_name][dim] = hub_score_dict
        std_N_k_results_no_norm.append(N_k_result)    
        
        
        std_normal_normed = preprocessing.normalize(std_normal, norm = 'l2', axis = 1)
        N_k_result, hub_score_dict = analysis.get_k_occurrence_and_hubness_score_dict(
            std_normal_normed, k)
        
        k_skew_and_rh_dict[std_normal_norm_name][dim] = hub_score_dict
        std_N_k_results.append(N_k_result)  
    
    std_no_norm_title = 'a) ' # std normal
    std_norm_title = 'b)' #  std normal normed emb
    
    
    f_dist_name = 'f_dist'
    f_dist_force_n_name = 'f_dist_forced_normal'
    
    k_skew_and_rh_dict[f_dist_name] = {}
    k_skew_and_rh_dict[f_dist_force_n_name] = {}
    
    f_dist_data = []
    N_k_results_f_dist = []
     
    for dim in dimensions:
        _, _, f_dist_various_means = synthetic_data.make_f_distributed_data(dim, rng)
        f_dist_data.append(f_dist_various_means)
        
        N_k_result, hub_score_dict = analysis.get_k_occurrence_and_hubness_score_dict(
            f_dist_various_means, k)
        
        k_skew_and_rh_dict[f_dist_name][dim] = hub_score_dict
        N_k_results_f_dist.append(N_k_result)
    
    
    f_dist_title = 'c)' # F dist
    
    
    N_k_results_force_norm = []
    
    for i in range(len(dimensions)):
        f_dist_various_means_norm = reduction.transform_dimensions(
            f_dist_data[i],
            random_state,
            reduction.ProbabilityDistribution.NORMAL,
            normalize_rows = True)
        
        N_k_result, hub_score_dict = analysis.get_k_occurrence_and_hubness_score_dict(
            f_dist_various_means_norm, k)
        
        k_skew_and_rh_dict[f_dist_force_n_name][dimensions[i]] = hub_score_dict
        N_k_results_force_norm.append(N_k_result)        
    
    
    f_dist_norm_title = 'd)' # F dist forced normal
    
    
    N_k_results = [std_N_k_results_no_norm, std_N_k_results, N_k_results_f_dist, N_k_results_force_norm]
    sub_titles = [std_no_norm_title, std_norm_title, f_dist_title, f_dist_norm_title]
    dimensions_strings = [str(d) for d in dimensions]
    
    fig, axes = analysis.horizontally_stacked_N_k_plots_dims_in_same(
            N_k_arrays = N_k_results,
            title = 'Distributions of k-occurrence',
            sub_titles = sub_titles,
            dimensions = dimensions_strings,
            width = 20,
            height = 5)
    
    save_to_file.save_pdf_fig('combined_N_k_figure_article', save_folder)
    
    hubness_scores_path = os.path.join(save_folder, 'synthetic_data_hubness_scores_abbreviated_article.json')
    save_to_file.save_as_json(k_skew_and_rh_dict, save_to = hubness_scores_path)
    

def synthetic_data_plots_for_article_full():
    random_state = 0
    rng = default_rng(random_state)
    
    save_folder = 'output/figures'
    
    k = 10
    dimensions = [3, 20, 768]
    dim_str = '_'.join(map(str, dimensions))
    
    f_dist_name = 'f_dist'
    f_dist_force_n_name = 'f_dist_forced_normal'
    
    k_skew_and_rh_dict = {f_dist_name : {}, f_dist_force_n_name: {} }
    
    f_dist_data = []
    N_k_results = []
     
    for dim in dimensions:
        _, _, f_dist_various_means = synthetic_data.make_f_distributed_data(dim, rng)
        f_dist_data.append(f_dist_various_means)
        
        N_k_result, hub_score_dict = analysis.get_k_occurrence_and_hubness_score_dict(
            f_dist_various_means, k)
        
        k_skew_and_rh_dict[f_dist_name][dim] = hub_score_dict
        N_k_results.append(N_k_result)
    
    
    fig, axes = analysis.horizontally_stacked_N_k_plots_for_dimensions(
        dimensions = dimensions,
        N_k_results = N_k_results,
        title = 'Data F distributed in each dimension ',
        always_log = True)
    
    save_to_file.save_pdf_fig(f'{f_dist_name}_{dim_str}', save_folder)
    
    
    N_k_results_force_norm = []
    
    for i in range(len(dimensions)):
        f_dist_various_means_norm = reduction.transform_dimensions(
            f_dist_data[i],
            random_state,
            reduction.ProbabilityDistribution.NORMAL,
            normalize_rows = True)
        
        N_k_result, hub_score_dict = analysis.get_k_occurrence_and_hubness_score_dict(
            f_dist_various_means_norm, k)
        
        k_skew_and_rh_dict[f_dist_force_n_name][dimensions[i]] = hub_score_dict
        N_k_results_force_norm.append(N_k_result)        
    
    
    fig, axes = analysis.horizontally_stacked_N_k_plots_for_dimensions(
        dimensions = dimensions,
        N_k_results = N_k_results_force_norm,
        title = 'Data F distributed forced normal ',
        always_log = True)
    
    save_to_file.save_pdf_fig(f'{f_dist_force_n_name}_{dim_str}', save_folder)
        
    
    std_normal_name = 'std_normal_no_norm'
    std_normal_norm_name = 'std_normal_norm_emb'
    uni_name = 'uniform_no_norm'
    uni_norm_name = 'uniform_norm_emb'
    normal_mean_1_name = 'normal_mean_1'
    normal_v_means_name = 'normal_v_means'
    
    k_skew_and_rh_dict[std_normal_name] = {}
    k_skew_and_rh_dict[std_normal_norm_name] = {}
    k_skew_and_rh_dict[uni_name] = {}
    k_skew_and_rh_dict[uni_norm_name] = {}
    k_skew_and_rh_dict[normal_mean_1_name] = {}
    k_skew_and_rh_dict[normal_v_means_name] = {} 
   
    normal_mean_1_dims = []
    
    std_N_k_results_no_norm = []
    std_N_k_results = []
    uni_N_k_results_no_norm = []
    uni_N_k_results = []
    mean_1_N_k_results = []
    v_means_N_k_results = []
    
    
    for dim in dimensions:
        std_normal, normal_mean_1, various_means = synthetic_data.make_normal_dist_data(dim, rng)
                
        normal_mean_1_dims.append(normal_mean_1) 
        
        N_k_result, hub_score_dict = analysis.get_k_occurrence_and_hubness_score_dict(
            std_normal, k)
        
        k_skew_and_rh_dict[std_normal_name][dim] = hub_score_dict
        std_N_k_results_no_norm.append(N_k_result)    
        
        
        std_normal_normed = preprocessing.normalize(std_normal, norm = 'l2', axis = 1)
        N_k_result, hub_score_dict = analysis.get_k_occurrence_and_hubness_score_dict(
            std_normal_normed, k)
        
        k_skew_and_rh_dict[std_normal_norm_name][dim] = hub_score_dict
        std_N_k_results.append(N_k_result)  
        
        
        mean_1_normed = preprocessing.normalize(normal_mean_1, norm = 'l2', axis = 1)
        N_k_result, hub_score_dict = analysis.get_k_occurrence_and_hubness_score_dict(
            mean_1_normed, k)
        
        k_skew_and_rh_dict[normal_mean_1_name][dim] = hub_score_dict
        mean_1_N_k_results.append(N_k_result)  
        
        
        v_means_normed = preprocessing.normalize(various_means, norm = 'l2', axis = 1)
        N_k_result, hub_score_dict = analysis.get_k_occurrence_and_hubness_score_dict(
            v_means_normed, k)
        
        k_skew_and_rh_dict[normal_v_means_name][dim] = hub_score_dict
        v_means_N_k_results.append(N_k_result)
        
        
        uni_dist = synthetic_data.make_uni_dist_data_mean_0(dim, rng)
        
        N_k_result, hub_score_dict = analysis.get_k_occurrence_and_hubness_score_dict(
            uni_dist, k)
        
        k_skew_and_rh_dict[uni_name][dim] = hub_score_dict
        uni_N_k_results_no_norm.append(N_k_result)    
        
        
        uni_normed = preprocessing.normalize(uni_dist, norm = 'l2', axis = 1)
        N_k_result, hub_score_dict = analysis.get_k_occurrence_and_hubness_score_dict(
            uni_normed, k)
        
        k_skew_and_rh_dict[uni_norm_name][dim] = hub_score_dict
        uni_N_k_results.append(N_k_result)  
        
        
        
    fig, axes = analysis.horizontally_stacked_N_k_plots_for_dimensions(
        dimensions = dimensions,
        N_k_results = std_N_k_results_no_norm,
        title = 'Data std normal distributed in each dimension ',
        always_log = True)
    
    save_to_file.save_pdf_fig(f'{std_normal_name}_{dim_str}', save_folder)
    
    
    fig, axes = analysis.horizontally_stacked_N_k_plots_for_dimensions(
        dimensions = dimensions,
        N_k_results = std_N_k_results,
        title = 'Data std normal dist in each dimension - normed emb ',
        always_log = True)
    
    save_to_file.save_pdf_fig(f'{std_normal_norm_name}_{dim_str}', save_folder)
    
    
    fig, axes = analysis.horizontally_stacked_N_k_plots_for_dimensions(
        dimensions = dimensions,
        N_k_results = mean_1_N_k_results,
        title = 'Data normal distributed in each dimension mean 1 ',
        always_log = True)
    
    save_to_file.save_pdf_fig(f'{normal_mean_1_name}_{dim_str}', save_folder)
    
    
    fig, axes = analysis.horizontally_stacked_N_k_plots_for_dimensions(
        dimensions = dimensions,
        N_k_results = v_means_N_k_results,
        title = 'Data normal distributed in each dimension various means ',
        always_log = True)
    
    save_to_file.save_pdf_fig(f'{normal_v_means_name}_{dim_str}', save_folder)
    
    
    normal_mean_1_centered_name = 'normal_mean_1_centered'
    
    k_skew_and_rh_dict[normal_mean_1_centered_name] = {}
    
    mean_1_cen_N_k_results = []
    
    for i in range(len(dimensions)):
        normal_mean_1_centered = reduction.np_center_vectors(normal_mean_1_dims[i])
                
        mean_1_centered_normed = preprocessing.normalize(normal_mean_1_centered, norm = 'l2', axis = 1)
        N_k_result, hub_score_dict = analysis.get_k_occurrence_and_hubness_score_dict(
            mean_1_centered_normed, k)
        
        k_skew_and_rh_dict[normal_mean_1_centered_name][dimensions[i]] = hub_score_dict
        mean_1_cen_N_k_results.append(N_k_result)
    
    fig, axes = analysis.horizontally_stacked_N_k_plots_for_dimensions(
        dimensions = dimensions,
        N_k_results = mean_1_cen_N_k_results,
        title = 'Data normal distributed mean 1 centered and normalised ',
        always_log = True)
    
    save_to_file.save_pdf_fig(f'{normal_mean_1_centered_name}_{dim_str}', save_folder)
    
    
    fig, axes = analysis.horizontally_stacked_N_k_plots_for_dimensions(
        dimensions = dimensions,
        N_k_results = uni_N_k_results_no_norm,
        title = 'Data uniformly distributed in each dimension [-1, 1] ',
        always_log = True)
    
    save_to_file.save_pdf_fig(f'{uni_name}_{dim_str}', save_folder)
    
    
    fig, axes = analysis.horizontally_stacked_N_k_plots_for_dimensions(
        dimensions = dimensions,
        N_k_results = uni_N_k_results,
        title = 'Data uniformly distributed in each dimension [-1, 1] - normed emb ',
        always_log = True)
    
    save_to_file.save_pdf_fig(f'{uni_norm_name}_{dim_str}', save_folder)
    
    
    hubness_scores_path = os.path.join(save_folder, 'synthetic_data_hubness_scores_article.json')
    save_to_file.save_as_json(k_skew_and_rh_dict, save_to = hubness_scores_path)
    
    
def k_occurrence_dist_plot_article(
        result_folder_path: str = 'output/r_20_st'):
    
    final_results.make_k_occurrence_dist_plot(
        result_folder_path = result_folder_path,
        dataset = 'newsgroups',
        base_model = 'distilroberta-base',
        seed = 1,
        reductions = ['base', 'normal_all', 'mutual_proximity', 'normal_all_mutual_proximity'],
        dist_measure = 'euclidean', 
        train_norm = 'n',
        subset = 'train',
        width = 20,
        height = 5,
        file_name_suffix = 'article',
        plot_vs_std_norm = True)


def base_vs_reductions_plots_for_article(
        output_folder: str, 
        dataset: str,
        prefix: str = '',
        subsets: List[str] = ['test'],
        save_folder: str = 'output/figures',
        results_folder: str = '') -> None:
        
    pair_df, _ = final_results.get_data_for_paired_comparison(
        output_folder,
        dataset,
        reductions = [],
        prefix = prefix,
        subsets = subsets,
        results_folder = results_folder)
    
    base_vs_reductions_plots_for_article_from_df(
        pair_df,
        dataset,
        prefix,
        save_folder)


def base_vs_reductions_plots_example_article():
    output_folder = 'output'
    dataset = 'newsgroups'
    prefix = 'sts_bert_distilroberta-base'
    subsets = ['test']
    save_folder = 'output/figures'
    results_folder = 'r_20_st'
    
    base_vs_reductions_plots_for_article(
            output_folder, 
            dataset,
            prefix,
            subsets,
            save_folder,
            results_folder)
    
    prefix = 'sts_bert_microsoft-MiniLM-L12-H384-uncased'
    prefix = 'sts_bert_microsoft-mpnet'
    
    dataset = 'ag_news'
    results_folder = 'r_ag'
    
    dataset = 'yahoo_answers_small'
    results_folder = 'r_ya'
    
    # TODO: Generate all plots



def plot_dimension_dist_examples(
        embeddings: NDArray, 
        example_dims: List[int],
        bins: int,
        title: str,
        model_name: str,
        width: int = 15,
        height: int = 7.5):
    
    transposed_emb = np.transpose(embeddings)
    dims = transposed_emb[example_dims]
    min_val = dims.min()
    max_val =  dims.max()
    
    scale = height/7.5
    
    fig, axes = plt.subplots(
        1, ncols = len(example_dims), 
        figsize=(width,height),
        sharey = True)
    fig.suptitle(title, fontsize = 34 * scale)
    
    for i in range(len(example_dims)):   
        n, bins, patches = axes[i].hist(
            dims[i], 
            bins, 
            range = (min_val, max_val),
            density=False, facecolor='g', alpha=0.75)
        
        axes[i].grid(True)
        axes[i].set_xlim([min_val, max_val])
        
        sub_title = f'dimension {example_dims[i]}'
        if i == (len(example_dims) - 1)/2:
            sub_title = f'{model_name}\n' + sub_title
        
        axes[i].set_title(sub_title, fontsize = 28 * scale)
        axes[i].tick_params(axis='both', which='both', labelsize = 20 * scale) 
        
    axes[0].set_ylabel('Number of embeddings', fontsize = 28 * scale)
    fig.text(0.51, -0.03, 'Value', ha='center', fontsize = 28 * scale)
    fig.tight_layout()
        
    return fig, axes


def plot_distribution_of_dimension_values():
    output_folder = 'output'
    fig_folder = 'figures'
    save_folder = os.path.join(output_folder, fig_folder)
    embedding_folder = 'embeddings_20_newsgroups_self_trained'
    
    example_dimensions = [0, 5, 6]

    example_embedding_file = 'sts_bert_distilroberta-base_cos_dist_ORTHOGONAL_z_False_n_False_c_False_seed1_newsgroups_train.h5'
    
    load_path = os.path.join(output_folder,embedding_folder, example_embedding_file)
    embeddings = sent_bert.load_sentence_bert_embeddings(load_path)
       
    fig, axes = plot_dimension_dist_examples(
        embeddings = embeddings,
        example_dims = example_dimensions,
        bins = 100,
        title = '20 newsgroups dimension distribution example of embeddings',
        model_name = 'distilroberta-base_cos_dist_none_seed1')
    
    save_to_file.save_pdf_fig('distilroberta-base_cos_dist_none_seed1_newsgroups_dim_dist_example', save_folder)
    
    
    embedding_folder = 'embeddings'

    example_embedding_file = 'sts_bert_distilroberta-base_cos_dist_ORTHOGONAL_z_False_n_False_c_False_seed1_ag_news_train.h5'
    
    load_path = os.path.join(output_folder,embedding_folder, example_embedding_file)
    embeddings = sent_bert.load_sentence_bert_embeddings(load_path)
            
    fig, axes = plot_dimension_dist_examples(
        embeddings = embeddings,
        example_dims = example_dimensions,
        bins = 100,
        title = 'AG News dimension distribution example of embeddings',
        model_name = 'distilroberta-base_cos_dist_none_seed1')
    
    save_to_file.save_pdf_fig('distilroberta-base_cos_dist_none_seed1_ag_news_dim_dist_example', save_folder)
        


def build_and_save_increased_acc_decrease_hubness_tables(
        results_folders: List[str] = ['r_20_st', 'r_ag', 'r_ya']):
    
    datasets = ['newsgroups', 'ag_news', 'yahoo_answers_small']
    
    output_folder = 'output'    
    subset = 'test'
    save_folder = 'output/figures'
    
    for i, r_folder in enumerate(results_folders):
        dataset = datasets[i]
        
        final_results.build_and_save_increased_acc_decreased_hubness_table(
            output_folder,
            dataset,
            subset,
            save_folder,
            r_folder)


def build_and_save_pretrained_model_tables():
    save_folder = 'output/figures'
    
    results_folder_path = 'output/results_20_newsgroups_pretrained'
    dataset = 'newsgroups'
    
    pair_pretrain_df = final_results.get_data_for_pretrained_models_table(
        results_folder_path, dataset)
    
    fig,ax = final_results.build_table(
        pair_pretrain_df.astype(float).round(decimals = 3), 
        pair_pretrain_df.index,
        header_columns=0, 
        col_width=5.0)
    save_to_file.save_pdf_fig(f'{dataset}_pretrained_models_acc_hubness', save_folder)
    
    
    results_folder_path = 'output/results_ag_news_pretrained'
    dataset = 'ag_news'
    
    pair_pretrain_ag_df = final_results.get_data_for_pretrained_models_table(
        results_folder_path, dataset)
    
    fig,ax = final_results.build_table(
        pair_pretrain_ag_df.astype(float).round(decimals = 3), 
        pair_pretrain_ag_df.index,
        header_columns=0, 
        col_width=5.0)
    save_to_file.save_pdf_fig(f'{dataset}_pretrained_models_acc_hubness', save_folder)
    
    
    results_folder_path = 'output/results_yahoo_answers_pretrained'
    dataset = 'yahoo_answers_small'
    
    pair_pretrain_ya_df = final_results.get_data_for_pretrained_models_table(
        results_folder_path, dataset)
    
    fig,ax = final_results.build_table(
        pair_pretrain_ya_df.astype(float).round(decimals = 3), 
        pair_pretrain_ya_df.index,
        header_columns=0, 
        col_width=5.0)
    save_to_file.save_pdf_fig(f'{dataset}_pretrained_models_acc_hubness', save_folder)


def build_and_save_overview_tables():
    output_folder = 'output'
    subsets = ['test']
    save_suffix = 'overview_table'
    
    
    base_models = ['microsoft-mpnet-base', 'distilroberta-base', 'microsoft-MiniLM-L12-H384-uncased']
    
    
    dataset = 'newsgroups'
    results_folder = 'r_20_st'
    
    for base_model in base_models:
        final_results.build_and_save_overview_table_from_results(
                output_folder = output_folder,
                dataset = dataset,
                save_as = f'figures/{dataset}_{base_model}_{save_suffix}',
                subsets = subsets,
                prefix = f'sts_bert_{base_model}',
                results_folder = results_folder,
                use_error_rate = True)
    
    dataset = 'ag_news'
    results_folder = 'r_ag'
    
    for base_model in base_models[:2]:
        final_results.build_and_save_overview_table_from_results(
                output_folder = output_folder,
                dataset = dataset,
                save_as = f'figures/{dataset}_{base_model}_{save_suffix}',
                subsets = subsets,
                prefix = f'sts_bert_{base_model}',
                results_folder = results_folder,
                use_error_rate = True)
        
    dataset = 'yahoo_answers_small'
    results_folder = 'r_ya'
    
    for base_model in base_models[:2]:
        final_results.build_and_save_overview_table_from_results(
                output_folder = output_folder,
                dataset = dataset,
                save_as = f'figures/{dataset}_{base_model}_{save_suffix}',
                subsets = subsets,
                prefix = f'sts_bert_{base_model}',
                results_folder = results_folder,
                use_error_rate = True)
    
    
def make_none_vs_norm_plots():
    
    output_folder = 'output'
    subsets = ['test']
    save_folder = 'output/figures'
    
    reductions = ['base', 'normal_all_mutual_proximity']
    
    dataset = 'newsgroups'
    results_folder = 'r_20_st'
    
    
    # final_results.make_distance_measure_plots_none_vs_norm(
    #         output_folder = output_folder, 
    #         dataset = dataset,
    #         reductions = reductions,
    #         prefix = '',
    #         subsets = subsets,
    #         results_folder = results_folder,
    #         save_folder = save_folder)
    
    
    final_results.make_distance_measure_plots_none_vs_norm_base_vs_comb(
        output_folder = output_folder, 
        dataset = dataset,
        reductions = (reductions[0], reductions[1]),
        prefix = '',
        subsets = subsets,
        results_folder = results_folder,
        save_folder = save_folder)
    
        
    dataset = 'ag_news'
    results_folder = 'r_ag'
    
    # final_results.make_distance_measure_plots_none_vs_norm(
    #         output_folder = output_folder, 
    #         dataset = dataset,
    #         reductions = reductions,
    #         prefix = '',
    #         subsets = subsets,
    #         results_folder = results_folder,
    #         save_folder = save_folder)
    
    final_results.make_distance_measure_plots_none_vs_norm_base_vs_comb(
        output_folder = output_folder, 
        dataset = dataset,
        reductions = (reductions[0], reductions[1]),
        prefix = '',
        subsets = subsets,
        results_folder = results_folder,
        save_folder = save_folder)
        
        
    dataset = 'yahoo_answers_small'
    results_folder = 'r_ya'
    
    # final_results.make_distance_measure_plots_none_vs_norm(
    #         output_folder = output_folder, 
    #         dataset = dataset,
    #         reductions = reductions,
    #         prefix = '',
    #         subsets = subsets,
    #         results_folder = results_folder,
    #         save_folder = save_folder)
    
    final_results.make_distance_measure_plots_none_vs_norm_base_vs_comb(
        output_folder = output_folder, 
        dataset = dataset,
        reductions = (reductions[0], reductions[1]),
        prefix = '',
        subsets = subsets,
        results_folder = results_folder,
        save_folder = save_folder)    
    
    
def make_before_after_spy_plot():
    save_folder = 'output/figures'
    
    # Accuracy goes down when using force normal on this model
    model_name = 'sts_bert_microsoft-mpnet-base_cos_ORTHOGONAL_z_False_n_False_c_False_seed7'
    dataset = 'newsgroups'
    subset = 'test'
    
    embedding_folder = 'embeddings_20_newsgroups_self_trained'
    embedding_folder_path = os.path.join('output', embedding_folder)
    
    title = 'Spy plot of embeddings'
    
    fig, axes = visualization.get_before_after_spyplot(
        model_name = model_name,
        dataset = dataset,
        subset = subset,
        embedding_folder_path = embedding_folder_path,
        title = title,
        reduction = 'normal_all')
    
    save_to_file.save_high_res_png(f'spy_before_after_{dataset}_{subset}_{model_name}', save_folder)
    
    
    model_name = 'sts_bert_microsoft-MiniLM-L12-H384-uncased_cos_ORTHOGONAL_z_False_n_True_c_False_seed1'
    dataset = 'newsgroups'
    subset = 'test'
    
    embedding_folder = 'embeddings_20_newsgroups_self_trained'
    embedding_folder_path = os.path.join('output', embedding_folder)
    
    title = 'Spy plot of embeddings'
    
    fig, axes = visualization.get_before_after_spyplot(
        model_name = model_name,
        dataset = dataset,
        subset = subset,
        embedding_folder_path = embedding_folder_path,
        title = title,
        reduction = 'normal_all')
    
    save_to_file.save_high_res_png(f'spy_before_after_{dataset}_{subset}_{model_name}', save_folder)
    
 










