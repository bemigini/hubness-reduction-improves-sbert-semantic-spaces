# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:22:56 2022

@author: bmgi


Making figures for poster

TODO: Update to use the same folders


"""


from numpy.random import default_rng

import os
import pandas as pd

from sklearn import preprocessing

from src.hubness import analysis, synthetic_data
from src.performance import final_results
from src.util import save_to_file


def base_vs_reductions_plots_for_poster(
        pair_df: pd.DataFrame,
        dataset: str,
        save_folder: str = 'output/figures') -> None:
    
    if dataset == 'newsgroups':
        base_models = ['distilroberta-base', 'microsoft-mpnet-base']
    else:
        base_models = ['microsoft-mpnet-base']
    
    dist_measure = 'euclidean'
    train_norm = 'n'
    
    for base_model in base_models:
            fig, axes = final_results.horizontally_stacked_paired_plots(                
                pair_df = pair_df,
                dataset = dataset,
                base_model = base_model, 
                dist_measure = dist_measure, 
                train_norm = train_norm,
                width = 30,
                height = 7.5)
            
            file_name = f'{base_model}_{dist_measure}_{train_norm.replace(", ", "_")}_base_vs_reductions_{dataset}_poster'
            save_to_file.save_pdf_fig(file_name, save_folder)
                

def synthetic_data_plots_for_poster():
    save_folder = 'output/figures/'
    random_state = 0
    rng = default_rng(random_state)
    
    k = 10
    dimensions = [3, 20, 768]
    
    std_normal_name = 'std_normal_no_norm'
    std_normal_norm_name = 'std_normal_norm_emb'
    
    k_skew_and_rh_dict = {std_normal_name : {}, std_normal_norm_name: {} }
    
    std_N_k_results_no_norm = []
    std_N_k_results = []    
    
    
    std_normal_no_norm_title = 'K-occurrence distribution - std normal dist'
    std_normal_normed_title = 'K-occurrence distribution - std normal dist - emb unit length'
    
    for dim in dimensions:
        std_normal, normal_mean_1, various_means = synthetic_data.make_normal_dist_data(dim, rng)
                
        N_k_result, hub_score_dict = synthetic_data.get_k_occurrence_and_hubness_score_dict(
            std_normal, k)
        
        k_skew_and_rh_dict[std_normal_name][dim] = hub_score_dict
        std_N_k_results_no_norm.append(N_k_result)
        
        
        std_normal_normed = preprocessing.normalize(std_normal, norm = 'l2', axis = 1)
        
        N_k_result, hub_score_dict = synthetic_data.get_k_occurrence_and_hubness_score_dict(
            std_normal_normed, k)
        
        k_skew_and_rh_dict[std_normal_norm_name][dim] = hub_score_dict
        std_N_k_results.append(N_k_result)
    
    
    fig, axes = analysis.horizontally_stacked_N_k_plots_for_dimensions(
        dimensions = dimensions,
        N_k_results = std_N_k_results_no_norm,
        title = std_normal_no_norm_title,
        always_log = True,
        width = 30,
        height = 7.5)
    
    file_name = f'{std_normal_name}_poster'
    save_to_file.save_pdf_fig(file_name, save_folder)
    
    
    fig, axes = analysis.horizontally_stacked_N_k_plots_for_dimensions(
        dimensions = dimensions,
        N_k_results = std_N_k_results,
        title = std_normal_normed_title,
        always_log = True,
        width = 30,
        height = 7.5)
    
    file_name = f'{std_normal_norm_name}_poster'    
    save_to_file.save_pdf_fig(file_name, save_folder)
    
    hubness_scores_path = os.path.join(save_folder, 'synthetic_data_hubness_scores_poster.json')
    save_to_file.save_as_json(k_skew_and_rh_dict, save_to = hubness_scores_path)
    
    
    
def k_occurrence_dist_plot_poster(
        result_folder_path: str = 'output/results_20_newsgroups_self_trained'):
    
    final_results.make_k_occurrence_dist_plot(
        result_folder_path = result_folder_path,
        dataset = 'newsgroups',
        base_model = 'distilroberta-base',
        seed = 1,
        reductions = ['base', 'normal_all', 'mutual_proximity', 'normal_all_mutual_proximity'],
        dist_measure = 'euclidean', 
        train_norm = 'n',
        subset = 'train',
        width = 30,
        height = 7.5,
        file_name_suffix = 'poster')


    
   
    






