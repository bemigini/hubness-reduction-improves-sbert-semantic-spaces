# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 09:22:44 2022

@author: bmgi


Post training reduction methods


"""


from enum import Enum

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from scipy.sparse import csr_matrix

import skhubness as skh
import sklearn

from typing import Tuple


class ProbabilityDistribution(Enum):
    NORMAL = 'normal'
    UNIFORM = 'uniform'


class ForceDistribution(Enum):
    NONE = 'none'
    NORMAL_ALL = 'normal_all'
    NORMAL_SPLITS = 'normal_splits'
    UNIFORM_ALL = 'uniform_all'
    UNIFORM_SPLITS = 'uniform_splits'

class SkhubnessReduction(Enum):
    NONE = 'none'
    MUTUAL_PROXIMITY = 'mutual_proximity'
    LOCAL_SCALING = 'local_scaling'
    

def transform_to_dist_in_each_dim(
        vectors: NDArray, 
        dist_in_vector_shape: NDArray) -> NDArray:
    
    v_shape = vectors.shape
    
    dist_vectors = np.zeros(v_shape)
    dimensions = v_shape[1]
    for d in range(dimensions):
        dim_d = vectors[:, d]
        dim_d_normal = dist_in_vector_shape[:, d]
        sorted_idx_dim_d = np.argsort(dim_d)
        sorted_idx_dim_d_normal = np.argsort(dim_d_normal)
        dist_vectors[sorted_idx_dim_d, d] = dim_d_normal[sorted_idx_dim_d_normal]
        
    return dist_vectors


def transform_dimensions(
        vectors: NDArray, 
        random_state: int,
        distribution_type: ProbabilityDistribution,
        normalize_rows: bool):
    
    rng = default_rng(random_state)
    v_shape = vectors.shape
    if distribution_type == ProbabilityDistribution.NORMAL:
        distribution = rng.standard_normal(v_shape)
    elif distribution_type == ProbabilityDistribution.UNIFORM:
        distribution = rng.uniform(low = -1, high = 1, size = v_shape)
    else:
        raise ValueError('Probability distribution not recognized')
        
    dist_vectors = transform_to_dist_in_each_dim(vectors, distribution)
    
    if normalize_rows:        
        dist_vectors = sklearn.preprocessing.normalize(
            dist_vectors,
            norm = 'l2',
            axis = 1, 
            return_norm = False)
    
    return dist_vectors


def transform_all_embeddings(
        train_embeddings: NDArray,
        test_embeddings: NDArray,
        force_dist: ForceDistribution,
        random_state: int):
    num_train_embeddings = train_embeddings.shape[0]
    all_embeddings = np.concatenate((train_embeddings, test_embeddings))
    
    if force_dist == ForceDistribution.NORMAL_ALL:
        all_embeddings_dist_norm = transform_dimensions(
            all_embeddings, 
            random_state,
            distribution_type = ProbabilityDistribution.NORMAL,
            normalize_rows = True)
    else:
        all_embeddings_dist_norm = transform_dimensions(
            all_embeddings, 
            random_state,
            distribution_type = ProbabilityDistribution.UNIFORM,
            normalize_rows = True)
    
    train_embeddings_dist = all_embeddings_dist_norm[:num_train_embeddings]
    test_embeddings_dist = all_embeddings_dist_norm[num_train_embeddings:]
    
    return train_embeddings_dist, test_embeddings_dist


def get_mutual_proximity_graph(graph_train: csr_matrix):    
    mp_hub_reduction = skh.reduction.MutualProximity(method = 'normal')
    mp_hub_reduction.fit(graph_train)
    graph_train = mp_hub_reduction.transform(graph_train)
    return mp_hub_reduction, graph_train


def get_mutual_proximity_graphs(
        graph_train: csr_matrix, 
        graph_test: csr_matrix):
    mp_hub_reduction, graph_train = get_mutual_proximity_graph(graph_train)
    graph_test = mp_hub_reduction.transform(graph_test)
    
    return graph_train, graph_test


def get_local_scaling_graphs(
        graph_train: csr_matrix, 
        graph_test: csr_matrix):
    ls_hub_reduction = skh.reduction.LocalScaling(
        k = 5, # TODO: Justify this choice
        method = 'standard')
    ls_hub_reduction.fit(graph_train)
    graph_train = ls_hub_reduction.transform(graph_train)
    graph_test = ls_hub_reduction.transform(graph_test)
    
    return graph_train, graph_test


def np_center_vectors(vectors: NDArray):
    vector_means = np.mean(vectors, axis = 1)
    
    centered_vectors = vectors - vector_means.reshape(-1, 1)
    
    return centered_vectors


def np_center_dimensions(vectors: NDArray, return_means: bool = False):
    dimension_means = np.mean(vectors, axis = 0)
    
    centered_dimensions = vectors - dimension_means
    
    if return_means:
        return centered_dimensions, dimension_means
    
    return centered_dimensions


def np_center_dimensions_median(vectors: NDArray):
    dimension_medians = np.median(vectors, axis = 0)
    
    centered_dimensions = vectors - dimension_medians
    
    return centered_dimensions


def np_center_dimensions_mode_bins(vectors: NDArray):
    dimension_modes = []
    
    for c in range(vectors.shape[1]):        
        counts, bins = np.histogram(vectors[:, c], bins = 75)
        dimension_modes.append(bins[np.argmax(counts)])        
    
    centered_dimensions = vectors - np.array(dimension_modes).reshape(1, -1)
    
    return centered_dimensions


def center_dimensions_normalize_embeddings(
        train_emb: NDArray, 
        test_emb: NDArray) -> Tuple[NDArray, NDArray]:
    train_centered, dimension_means = np_center_dimensions(train_emb, return_means = True)
    train_embeddings_c_norm = sklearn.preprocessing.normalize(
        train_centered, 
        axis = 1)
    test_embeddings_c_norm = sklearn.preprocessing.normalize(
        test_emb - dimension_means, 
        axis = 1)
    
    return train_embeddings_c_norm, test_embeddings_c_norm


