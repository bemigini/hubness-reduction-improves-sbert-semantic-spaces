# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 08:08:45 2022

@author: bmgi


Functions for exploring hubness


"""


from enum import Enum

import matplotlib.pyplot as plt

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from scipy.sparse import csr_matrix

import skhubness as skh

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsTransformer, NearestNeighbors

# from skhubness.neighbors import NMSlibTransformer
# Made custom code to handle when not all neighbours are found
from src.skhubness_custom.nmslib_wrap import NMSlibTransformer

from typing import Dict, List, Tuple
from tqdm import tqdm


class DistanceFunction(Enum):
    EUCLIDEAN = 'euclidean'
    NORM_EUCLIDEAN = 'norm_euclidean'
    COSINE = 'cosine'


def N_k_x(
        x_index: int, 
        nearest_k_for_every_point: NDArray) -> float:
    # If x is in the k nearest neighbours for a point it only occurs once
    # So we can sum all the occurences of x in the array of nearest neighbours. 
    among_k_nearest = (nearest_k_for_every_point == x_index).sum()
    
    return among_k_nearest
    

def get_N_k_from_k_neighbours_matrix(k_neighbours: NDArray, include_self: bool = True):
    N_k_counts = np.zeros(k_neighbours.shape[0])
    
    for i in range(k_neighbours.shape[0]):
        if include_self:
            current_neighbours = k_neighbours[i]
        else:
            current_neighbours = k_neighbours[i][k_neighbours[i] != i]
        N_k_counts[current_neighbours] = N_k_counts[current_neighbours] + 1
    
    return N_k_counts


def N_k_for_fitted(
        k: int, 
        vectors: NDArray, 
        fitted_n_neighbour: NearestNeighbors) -> NDArray:
    
    nearest_k = fitted_n_neighbour.kneighbors(
        vectors, 
        n_neighbors = k, 
        return_distance = False)
    
    num_vectors = vectors.shape[0]
    
    N_k_result = np.zeros(num_vectors).astype(int)
    for i in tqdm(range(num_vectors)):
        N_k_result[i] = N_k_x(
            x_index = i, 
            nearest_k_for_every_point = nearest_k)
    
    return N_k_result


def N_k(
        distance_function: DistanceFunction,
        k: int, 
        vectors: NDArray) -> NDArray:
    
    nn_vectors = np.copy(vectors)
    if distance_function == DistanceFunction.NORM_EUCLIDEAN:
        nn_vectors = preprocessing.normalize(nn_vectors, norm = 'l2', axis = 1)
        distance_name = 'euclidean'
    else:
        distance_name = distance_function.value
    
    nearest_neighbours = NearestNeighbors(n_neighbors = k, metric = distance_name)
    nearest_neighbours.fit(nn_vectors)
    
    N_k_result = N_k_for_fitted(k, nn_vectors, nearest_neighbours)    
    
    return N_k_result


def get_N_k_point_plot_data(
        N_k_array: NDArray, 
        as_prob: bool = False, 
        log_scale: bool = False) -> Tuple[NDArray, NDArray]:
    counts = np.bincount(N_k_array)
    non_zero_counts = counts[counts != 0]
    
    if as_prob:
        total = counts.sum()
        non_zero_counts = non_zero_counts / total
    
    bins = np.arange(counts.shape[0])
    non_zero_bins = bins[counts != 0]
    
    if log_scale:
        non_zero_counts = np.log10(non_zero_counts)
    
    return non_zero_bins, non_zero_counts


def get_N_k_point_plot(
        N_k_array: NDArray, 
        title: str, 
        as_prob: bool = False, 
        log_scale: bool = False,
        save_to: str = '') -> None:
    
    non_zero_bins, non_zero_counts = get_N_k_point_plot_data(
            N_k_array,
            as_prob,
            log_scale)
    
    plt.title(title)
    plt.plot(non_zero_bins, non_zero_counts, '-o')
    
    if save_to != '':
        plt.savefig(save_to)
    
    plt.show()


def plot_bins_counts_for_axis(
        non_zero_bins_dict: Dict[int, NDArray], 
        non_zero_counts_dict: Dict[int, NDArray],
        axis: object,
        dimensions: List[str],
        sub_title: str,
        y_range: List[float],
        x_range: List[float],
        scale: float,
        log_scale_x: bool,
        log_scale_y: bool):
    axis.set_ylim(y_range)
    axis.set_xlim(x_range)
    
    if log_scale_y:
        axis.set_yscale('log')
    if log_scale_x:
        axis.set_xscale('symlog')
    
    markers = ['o', 's', 'd', '+']
    colours = ['#377eb8', '#ff7f00', '#4daf4a']
    for i, k in enumerate(non_zero_bins_dict.keys()):
        marker = markers[i]
        linestyle = 'None' if dimensions[i] == '768' else '-'
        axis.plot(
            non_zero_bins_dict[k], 
            non_zero_counts_dict[k], 
            marker = marker,
            linestyle = linestyle,
            label = dimensions[i],
            color = colours[i])
        
    axis.set_title(sub_title, fontsize = 28 * scale)
    axis.tick_params(axis='both', which='both', labelsize = 20 * scale)
    
    return axis


def make_N_k_plots_dimensions_in_same(
        non_zero_bins_dicts: List[Dict[int, NDArray]],
        non_zero_counts_dicts: List[Dict[int, NDArray]],
        axes: plt.Axes,
        sub_titles: List[str],
        dimensions: List[str],
        log_scale_x: bool,
        log_scale_y: bool,
        scale: float):
    
    min_y = min([min(d[k]) 
                 for d in non_zero_counts_dicts
                 for k in d])
    max_y = max([max(d[k])
                 for d in non_zero_counts_dicts
                 for k in d])
    
    y_range = [max(min_y - 0.5/100000, 0.000001), max_y + 0.5/100]
    
    min_x = min([min(d[k]) 
                 for d in non_zero_bins_dicts 
                 for k in d])
    max_x = max([max(d[k]) 
                 for d in non_zero_bins_dicts 
                 for k in d])
    
    x_range = [min_x, max_x + 0.5]
    
    for j in range(len(non_zero_counts_dicts)):
        axes[j] = plot_bins_counts_for_axis(
                non_zero_bins_dict = non_zero_bins_dicts[j], 
                non_zero_counts_dict = non_zero_counts_dicts[j],
                axis = axes[j],
                dimensions = dimensions,
                sub_title = sub_titles[j],
                y_range = y_range,
                x_range = x_range,
                scale = scale,
                log_scale_x = log_scale_x,
                log_scale_y = log_scale_y)    
    
    axes[0].set_ylabel('Proportion', fontsize = 28 * scale)
    
    return axes


def horizontally_stacked_N_k_plots_dims_in_same(
        N_k_arrays: List[List[NDArray]],
        title: str,
        sub_titles: List[str],
        dimensions: List[str],
        width: float = 15,
        height: float = 7.5):    
    scale = height/7.5
    
    fig, axes = plt.subplots(
        1, ncols = len(N_k_arrays), figsize=(width,height), sharey = True)
    fig.suptitle(title, fontsize = 34 * scale, y = 1.0)
    
    non_zero_bins_dicts = []
    non_zero_counts_dicts = []
    
    for i in range(len(N_k_arrays)):
        non_zero_bins_dict = {}
        non_zero_counts_dict = {}
        for j in range(len(N_k_arrays[i])):            
            non_zero_bins, non_zero_counts = get_N_k_point_plot_data(
                    N_k_array = N_k_arrays[i][j],
                    as_prob = True,
                    log_scale = False)
            non_zero_bins_dict[j] = non_zero_bins
            non_zero_counts_dict[j] = non_zero_counts
            
        non_zero_bins_dicts.append(non_zero_bins_dict)
        non_zero_counts_dicts.append(non_zero_counts_dict)
    
    
    axes = make_N_k_plots_dimensions_in_same(
        non_zero_bins_dicts,
        non_zero_counts_dicts,
        axes,
        sub_titles,
        dimensions,
        log_scale_x = True,
        log_scale_y = True,
        scale = scale)
    
    fig.text(0.51, 0.0, 'N_k count', ha='center', fontsize = 28 * scale)
    legend = axes[-1].legend(title = 'dimensions', fontsize = 18 * scale)
    plt.setp(legend.get_title(), fontsize = 18 * scale)
    fig.tight_layout()
    
    return fig, axes


def make_N_k_plots_from_bins_and_counts(
        non_zero_bins_dict: Dict[int, NDArray], 
        non_zero_counts_dict: Dict[int, NDArray],
        axes: plt.Axes,
        sub_titles: List[str],
        log_scale_x: List[bool],
        log_scale_y: List[bool],
        scale: float,
        plot_vs_non_zero_bins: NDArray = np.array([]),
        plot_vs_non_zero_counts: NDArray = np.array([])):
    
    min_y = min([min(non_zero_counts_dict[k]) for k in non_zero_counts_dict])
    max_y = max([max(non_zero_counts_dict[k]) for k in non_zero_counts_dict])
    
    if plot_vs_non_zero_counts.any():
        min_plot_vs_counts = plot_vs_non_zero_counts.min()
        min_y = min(min_y, min_plot_vs_counts)
        max_plot_vs_counts = plot_vs_non_zero_counts.max()
        max_y = max(max_y, max_plot_vs_counts)
    
    y_range = [max(min_y - 0.5/100000, 0.000001), max_y + 0.5/100]
    
    min_x = min([min(non_zero_bins_dict[k]) for k in non_zero_bins_dict])
    max_x = max([max(non_zero_bins_dict[k]) for k in non_zero_bins_dict])
    
    if plot_vs_non_zero_bins.any():
        min_plot_vs_bins = plot_vs_non_zero_bins.min()
        min_y = min(min_y, min_plot_vs_bins)
        max_plot_vs_bins = plot_vs_non_zero_bins.max()
        max_y = max(max_y, max_plot_vs_bins)
    
    x_range = [min_x, max_x + 0.5]
    
    #blues = plt.get_cmap('Blues')
    
    for i in range(len(non_zero_bins_dict.keys())):
        if log_scale_y[i]:
            axes[i].set_yscale('log')
        if log_scale_x[i]:
            axes[i].set_xscale('symlog')
        
        axes[i].set_ylim(y_range)
        axes[i].set_xlim(x_range)
        
        axes[i].plot(
            non_zero_bins_dict[i], non_zero_counts_dict[i], 
            '-o', color = '#ff7f00', label = 'embedding')
        if plot_vs_non_zero_counts.any():            
            axes[i].plot(
                plot_vs_non_zero_bins, plot_vs_non_zero_counts, 
                '-o', color = '#377eb8', label = 'normal dist')                
            
        axes[i].set_title(sub_titles[i], fontsize = 28 * scale)
        axes[i].tick_params(axis='both', which='both', labelsize = 20 * scale)        
        
        if i > 0:
            axes[i].tick_params(axis='y', which='both', labelleft = False)
    
    axes[0].set_ylabel('Proportion', fontsize = 28 * scale)
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    return axes


def horizontally_stacked_N_k_plots_from_N_k_array(
        N_k_array: List[NDArray],
        title: str,
        sub_titles: List[str],
        log_scale_y: List[bool],
        log_scale_x: List[bool],
        always_log: bool = False,
        width: float = 15,
        height: float = 7.5,
        plot_vs_std_norm: bool = False,
        k: int = 10):
    
    if plot_vs_std_norm:
        random_state = 0
        rng = default_rng(random_state)
        dim = 768
        num_samples = N_k_array[0].shape[0]
        std_normal = rng.standard_normal((num_samples, dim))
        std_normal_normed = preprocessing.normalize(std_normal, norm = 'l2', axis = 1)
        std_N_k, _ = get_k_occurrence_and_hubness_score_dict(std_normal_normed, k)
        
        std_non_zero_bins, std_non_zero_counts = get_N_k_point_plot_data(
                N_k_array = std_N_k,
                as_prob = True,
                log_scale = False)
    else:
        std_non_zero_bins = np.array([])
        std_non_zero_counts = np.array([])    
    
    scale = height/7.5
    
    fig, axes = plt.subplots(1, ncols = len(N_k_array), figsize=(width,height))
    fig.suptitle(title, fontsize = 34 * scale)
    
    non_zero_bins_dict = {}
    non_zero_counts_dict = {}    
    
    for i in range(len(N_k_array)):        
        non_zero_bins, non_zero_counts = get_N_k_point_plot_data(
                N_k_array = N_k_array[i],
                as_prob = True,
                log_scale = False)
        non_zero_bins_dict[i] = non_zero_bins
        non_zero_counts_dict[i] = non_zero_counts
    
    
    axes = make_N_k_plots_from_bins_and_counts(
        non_zero_bins_dict,
        non_zero_counts_dict,
        axes,
        sub_titles,
        [always_log or log_scale_x[i] for i in range(len(N_k_array))],
        [always_log or log_scale_y[i] for i in range(len(N_k_array))],
        scale,
        std_non_zero_bins,
        std_non_zero_counts)
    
    fig.text(0.51, 0.0, 'N_k count', ha='center', fontsize = 28 * scale)
    if plot_vs_std_norm:
        axes[-1].legend(fontsize = 18 * scale, loc = 'upper right')
    fig.tight_layout()
    
    return fig, axes


def horizontally_stacked_N_k_plots_for_dimensions(
        dimensions: List[int],
        N_k_results: List[NDArray],
        title: str,
        log_scale_y: bool = True,
        log_scale_x: bool = True,
        always_log: bool = False,
        width: float = 20,
        height: float = 5):
    
    sub_titles = [f'{dimensions[i]} dimensions' for i in range(len(dimensions))]
    
    return horizontally_stacked_N_k_plots_from_N_k_array(
            N_k_array = N_k_results,
            title = title,
            sub_titles = sub_titles,
            log_scale_y = [log_scale_y and dimensions[i] > 20 for i in range(len(dimensions))],
            log_scale_x = [log_scale_x and dimensions[i] > 20 for i in range(len(dimensions))],
            always_log = always_log,
            width = width,
            height = height)
    
    
def get_N_k_and_plot(
        distance_function: DistanceFunction,
        k: int, 
        vectors: NDArray,
        title: str, 
        as_prob: bool = False, 
        log_scale: bool = False,
        save_to: str = '') -> NDArray:
    
    N_k_result =  N_k(distance_function, k, vectors)    
    
    get_N_k_point_plot(N_k_result, title, as_prob, log_scale, save_to)
    
    return N_k_result


def get_knn_transformation(
        neighbour_num: int,
        train_features: NDArray,
        output_folder: str,
        knn_metric: str = 'euclidean',
        knn_mode: str = 'distance',
        n_jobs: int = 5):
    
    if train_features.shape[0] > 50000:
        # Parameters as in https://github.com/nmslib/nmslib/blob/master/python_bindings/notebooks/search_vector_dense_optim.ipynb
        knn_trans = NMSlibTransformer(
            output_folder = output_folder,
            n_neighbors = neighbour_num + 1, 
            metric = knn_metric,
            method = 'hnsw',
            efConstruction = 100,
            M = 15,
            post_processing = 0,
            n_jobs = n_jobs)
    else:    
        # From scikit-learn user guide 1.6.6. Nearest Neighbors Transformer
        # https://scikit-learn.org/stable/modules/neighbors.html#neighbors-transformer
        # In KNeighborsTransformer we use the definition which includes each 
        # training point as its own neighbor in the count of n_neighbors. 
        # However, for compatibility reasons with other estimators which use the 
        # other definition, one extra neighbor will be computed when 
        # mode == 'distance'. To maximise compatibility with all estimators, a 
        # safe choice is to always include one extra neighbor in a custom nearest 
        # neighbors estimator, since unnecessary neighbors will be filtered by 
        # following estimators.
        knn_trans = KNeighborsTransformer(
            n_neighbors = neighbour_num + 1, 
            metric = knn_metric,
            mode = knn_mode)
    
    knn_trans.fit(train_features, None)
    
    return knn_trans


def get_k_hubness_scores(data: csr_matrix, k = 10, return_k_occurrence = True):
    hub = skh.Hubness(
        k = k, 
        return_value = 'all', 
        metric = 'euclidean',
        return_k_occurrence = return_k_occurrence)
    hub.fit(data)
    
    return hub.score()


def get_hubness_scores_for_NDArray(
        data: NDArray,
        neighbour_num: int,
        return_k_occurrence: bool,
        k: int = 10,
        knn_metric: str = 'euclidean',
        knn_mode: str = 'distance',
        output_folder: str = 'output',
        n_jobs: int = 5):
    
    knn_trans = get_knn_transformation(
        neighbour_num = neighbour_num,
        train_features = data,
        output_folder = output_folder,
        knn_metric = knn_metric,
        knn_mode = knn_mode,
        n_jobs = n_jobs)
    
    graph_data: csr_matrix = knn_trans.fit_transform(data, None)
    
    hub_scores = get_k_hubness_scores(
        graph_data, 
        k = k, 
        return_k_occurrence = return_k_occurrence)
    
    return hub_scores


def get_k_occurrence_and_hubness_score_dict(
        syn_data: NDArray,
        k: int):
    k_occurrence_and_hubs = get_hubness_scores_for_NDArray(
        data = syn_data, 
        neighbour_num = 50, 
        return_k_occurrence = True,
        k = k)
    
    hub_score_dict = {}
    hub_score_dict['k_skewness'] = k_occurrence_and_hubs['k_skewness']
    hub_score_dict['robinhood'] = k_occurrence_and_hubs['robinhood']
    
    N_k_result =  k_occurrence_and_hubs['k_occurrence']
    
    return N_k_result, hub_score_dict


def get_mode_fdist(dfnum: int, dfden: int):
    
    mode = ((dfnum - 2)/dfnum) * (dfden/(dfden + 2))
    
    return mode


def get_number_of_hubs_and_antihubs(N_k_result: NDArray):
    return (N_k_result > 11).sum(), (N_k_result == 1).sum()


