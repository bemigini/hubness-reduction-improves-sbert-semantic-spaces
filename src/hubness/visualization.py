# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:20:07 2022

@author: bmgi


Making plots to visualize hubness


"""




import matplotlib.pyplot as plt
import numpy as np

from sklearn import neighbors

from src import sentence_bert as sent_bert
from src.hubness import reduction as hub_reduction
from src.datasets import loading_datasets




def get_before_after_spyplot(
        model_name: str,
        dataset: str,
        subset: str,
        embedding_folder_path: str,
        title: str,
        reduction: str = 'normal_all',
        random_state: int = 0):
    n_neighbors = 10
    height = 8.5
    width = 17
    scale = height/7.5
    
    train_embeddings = sent_bert.load_subset_embeddings(
        model_name = model_name, 
        dataset = dataset, 
        subset = 'train',
        embedding_folder_path = embedding_folder_path)    
    
    if subset == 'test':
        test_embeddings = sent_bert.load_subset_embeddings(
            model_name = model_name, 
            dataset = dataset, 
            subset = 'test',
            embedding_folder_path = embedding_folder_path)
        
    if reduction == 'normal_all':
        reduction_name = 'f-norm'
        if subset == 'train':
            embeddings_before = train_embeddings
            embeddings_after = hub_reduction.transform_dimensions(
                train_embeddings, 
                random_state,
                distribution_type = hub_reduction.ProbabilityDistribution.NORMAL,
                normalize_rows = True)
        elif subset == 'test':
            embeddings_before = test_embeddings
            _, embeddings_after = hub_reduction.transform_all_embeddings(
                train_embeddings,
                test_embeddings,
                hub_reduction.ProbabilityDistribution.NORMAL,
                random_state = random_state)
        else:
            raise ValueError(f'Subset not recognized: {subset}')
    
    targets = loading_datasets.load_targets_from_dataset(
        dataset_folder = 'datasets',
        dataset = dataset, 
        subset = subset)
    
    sort_idx_targets = np.argsort(targets)
    
    sorted_before = embeddings_before[sort_idx_targets]
    sorted_after = embeddings_after[sort_idx_targets]
    
    
    neigh = neighbors.NearestNeighbors(n_neighbors = n_neighbors)
    neigh.fit(sorted_before)    
    n_graph_before = neigh.kneighbors_graph(
        X=None, n_neighbors=n_neighbors, mode='connectivity')
    
    neigh = neighbors.NearestNeighbors(n_neighbors = n_neighbors)
    neigh.fit(sorted_after)
    n_graph_after = neigh.kneighbors_graph(
        X=None, n_neighbors=n_neighbors, mode='connectivity')
    
    fig, axes = plt.subplots(nrows = 1, ncols = 2, sharey = True,
                             figsize=(width, height))
    fig.suptitle(title,
                 fontsize = 34 * scale, y = 1.0)   
    
    axes[0].spy(n_graph_before, precision=0, markersize = 0.01)
    axes[0].set_title(f'before {reduction_name}', fontsize = 28 * scale)
    axes[1].spy(n_graph_after, precision=0, markersize = 0.01)
    axes[1].set_title(f'after {reduction_name}', fontsize = 28 * scale)
    
    return fig, axes
    





