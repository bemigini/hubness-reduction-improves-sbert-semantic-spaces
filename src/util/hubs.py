# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 12:29:57 2022

@author: bmgi


Functions for finding hubs, hub distribution, bad hubs ect.


"""


from collections import Counter

import math
import matplotlib.pyplot as plt

import numpy as np
from numpy.typing import NDArray

from sklearn.neighbors import NearestNeighbors

from typing import List
from tqdm import tqdm



def N_k_x(
        fit_vector_index: int, 
        nearest_k: NDArray) -> float:
    
    # If x_index is in a row, it is only there once, since the same point will
    # not be counted as a neighbour twice.
    among_k_nearest_sum = (nearest_k == fit_vector_index).sum()
    
    return among_k_nearest_sum
    

def N_k(
        k: int, 
        fit_vectors: NDArray,
        test_vectors: NDArray, 
        fitted_n_neighbour: NearestNeighbors) -> NDArray:
    
    nearest_k = fitted_n_neighbour.kneighbors(
        test_vectors, 
        n_neighbors = k, 
        return_distance = False)
    
    num_vectors = fit_vectors.shape[0]
    
    N_k_result = np.zeros(num_vectors)
    for i in tqdm(range(num_vectors)):
        N_k_result[i] = N_k_x(
                    fit_vector_index = i, 
                    nearest_k = nearest_k)
    
    return N_k_result


def N_k_euc(
        k: int, 
        fit_vectors: NDArray,
        test_vectors: NDArray, 
        n_neighbours: int) -> List[float]:
    
    euclidean_nn = NearestNeighbors(n_neighbors = n_neighbours, 
                                    metric = 'euclidean')
    euclidean_nn.fit(fit_vectors)
    
    N_k_result = N_k(
        k = k, 
        fit_vectors = fit_vectors,
        test_vectors = test_vectors, 
        fitted_n_neighbour = euclidean_nn)
    
    return N_k_result


def get_log_count_maybe_zero(x: float) -> float:    
    return math.log(x + 1, 10)


# TODO: Make a plot where it is more clear
def get_N_k_plot(N_k_list: List[float], title: str, as_prob: bool = False, log_scale: bool = False):
    
    total = len(N_k_list)
    count_N_k = Counter(N_k_list)
    
    if as_prob:
        count_N_k = Counter({key: count_N_k[key]/total
                             for key in count_N_k.keys()})
        
    if log_scale:
        count_N_k_log = Counter({get_log_count_maybe_zero(key): count_N_k[key]
                             for key in count_N_k.keys()})
        count_N_k = count_N_k_log
        
    
    plt.title(title)
    plt.bar(count_N_k.keys(), count_N_k.values())
    N_k_bar = plt    
    
    return N_k_bar


# This corresponds to BN_k(x) from Radovanovic 2010, Hubs in Space
def get_bad_k_ocurrences_for_vector(
        fit_vector_index: int, 
        test_vectors: NDArray, 
        fit_labels: NDArray,
        test_labels: NDArray, 
        k: int,
        nearest_k: NDArray):
    
    among_k_nearest = (nearest_k == fit_vector_index).sum(axis = 1)
    # Labels for points where the vector is among the k nearest
    point_labels = test_labels[np.where(among_k_nearest)]
    
    v_label = fit_labels[fit_vector_index]
    bad_k_occurences = (point_labels != v_label).sum()
    
    return bad_k_occurences


def get_bad_k_ocurrences(
        test_vectors: NDArray, 
        fit_labels: NDArray,
        test_labels: NDArray, 
        k: int, 
        fitted_n_neighbour: NearestNeighbors):
    
    nearest_k = fitted_n_neighbour.kneighbors(
        test_vectors, 
        n_neighbors = k, 
        return_distance = False)
    
    num_vectors = fit_labels.shape[0]
    bad_k_occurences = np.zeros(num_vectors)
    for i in tqdm(range(num_vectors)):
        bad_k_occurences[i] = get_bad_k_ocurrences_for_vector(
            fit_vector_index = i, 
            test_vectors = test_vectors, 
            fit_labels = fit_labels, 
            test_labels = test_labels,
            k = k, 
            nearest_k = nearest_k)
        
    return bad_k_occurences
    
    
def get_bad_k_ocurrences_euc(
        fit_vectors: NDArray,
        test_vectors: NDArray, 
        fit_labels: NDArray,
        test_labels: NDArray, 
        k: int):
    
    euclidean_nn = NearestNeighbors(n_neighbors = k, 
                                    metric = 'euclidean')
    euclidean_nn.fit(fit_vectors)
    
    bad_k_occurences = get_bad_k_ocurrences(
        test_vectors = test_vectors,
        fit_labels = fit_labels,
        test_labels = test_labels,
        k = k, 
        fitted_n_neighbour = euclidean_nn)    
        
    return bad_k_occurences










