# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 10:28:42 2022

@author: bmgi


Hubness on synthetic data 


"""


import math

import numpy as np
from numpy.random import default_rng

import os

from src.hubness import analysis
from src.hubness import reduction

from typing import List




def make_data_with_varying_dimension_means(dim: int, rng: np.random.Generator):
        
    std_normal = rng.standard_normal((10000, dim))
    dimension_means = rng.uniform(low = -2, high = 2, size = dim)
    
    data_with_varying_means = std_normal
    for d in range(dim):
        data_with_varying_means[:, d] = data_with_varying_means[:, d] + dimension_means[d]
        
    return data_with_varying_means


def make_normal_dist_data(dim: int, rng: np.random.Generator):
    std_normal = rng.standard_normal((10000, dim))
    normal_mean_1 = std_normal + 1
    various_means = make_data_with_varying_dimension_means(dim, rng)
    
    return std_normal, normal_mean_1, various_means


def make_uni_dist_data_mean_0(dim: int, rng: np.random.Generator):
    uni_dist = rng.uniform(low = -1, high = 1, size = (10000, dim))
    
    return uni_dist


def plot_N_k_for_synthetic_data(
        k: int,
        dimensions: List[int], 
        dist_functions: List[analysis.DistanceFunction],
        save_to_folder: str,
        random_state: int = 0):
    rng = default_rng(random_state)
    
    for dim in dimensions:
        std_normal = rng.standard_normal((10000, dim))
        normal_mean_1 = std_normal + 1
        various_means = make_data_with_varying_dimension_means(dim, rng)
        uni = rng.uniform(low = -2, high = 2, size = (10000, dim))
        uni_mean_1 = uni + 1
        
        for dist in dist_functions:
            suffix = f', dim = {dim}, dist = {dist.value}'
            
            normal_title = f'Standard normal{suffix}'
            normal_mean1_title = f'Normal distribution mean 1, standard deviation 1{suffix}'
            various_means_title = f'Normal distributions varying means{suffix}'
            uniform_title = f'Uniform distribution -2 to 2{suffix}'
            uniform_mean1_title = f'Uniform distribution -1 to 3{suffix}'
            
            log_scale = False if dim < 50 else True
            
            normal_path = os.path.join(save_to_folder, f'std_normal_{dist.name}_dim{dim}.png')
            analysis.get_N_k_and_plot(
                dist, 
                k, 
                std_normal, 
                normal_title, 
                as_prob = True,
                log_scale = log_scale,
                save_to = normal_path)
            
            
            normal_mean1_path = os.path.join(save_to_folder, f'normal_mean1_{dist.name}_dim{dim}.png')
            analysis.get_N_k_and_plot(
                dist, 
                k, 
                normal_mean_1, 
                normal_mean1_title, 
                as_prob = True,
                log_scale = log_scale,
                save_to = normal_mean1_path)
            
            
            normal_varying_path = os.path.join(save_to_folder, f'normal_various_means_{dist.name}_dim{dim}.png')
            analysis.get_N_k_and_plot(
                dist, 
                k, 
                various_means, 
                various_means_title, 
                as_prob = True,
                log_scale = log_scale,
                save_to = normal_varying_path)
            
            
            uniform_path = os.path.join(save_to_folder, f'uniform_{dist.name}_dim{dim}.png')
            analysis.get_N_k_and_plot(
                dist, 
                k, 
                uni, 
                uniform_title, 
                as_prob = True,
                log_scale = log_scale,
                save_to = uniform_path)
            
            
            uniform_mean1_path = os.path.join(save_to_folder, f'uniform_mean1_{dist.name}_dim{dim}.png')
            analysis.get_N_k_and_plot(
                dist, 
                k, 
                uni_mean_1, 
                uniform_mean1_title, 
                as_prob = True,
                log_scale = log_scale,
                save_to = uniform_mean1_path)
            

def plot_centered_N_k_known_problem_data(
        k: int,
        dimensions: List[int], 
        dist_functions: List[analysis.DistanceFunction],
        save_to_folder: str,
        random_state: int = 0):
    rng = default_rng(random_state)
    
    for dim in dimensions:
        normal_mean_1 = rng.standard_normal((10000, dim)) + 1
        various_means = make_data_with_varying_dimension_means(dim, rng)
        uni_mean_1 = rng.uniform(low = -2, high = 2, size = (10000, dim)) + 1
        
        for dist in dist_functions:
            centering_functions = [reduction.np_center_vectors, reduction.np_center_dimensions]
            
            for func in centering_functions:                
                suffix = f', dim = {dim}, dist = {dist.value}, {func.__name__}'
                
                normal_mean1_title = f'Normal distribution mean 1, standard deviation 1{suffix}'
                various_means_title = f'Normal distributions varying means{suffix}'
                uniform_mean1_title = f'Uniform distribution -1 to 3{suffix}'
                
                log_scale = False if dim < 50 else True
                
                normal_mean1_path = os.path.join(save_to_folder, f'normal_mean1_{dist.name}_dim{dim}_{func.__name__}.png')
                analysis.get_N_k_and_plot(
                    dist, 
                    k, 
                    func(normal_mean_1), 
                    normal_mean1_title, 
                    as_prob = True,
                    log_scale = log_scale,
                    save_to = normal_mean1_path)
                
                
                normal_varying_path = os.path.join(save_to_folder, f'normal_various_means_{dist.name}_dim{dim}_{func.__name__}.png')
                analysis.get_N_k_and_plot(
                    dist, 
                    k, 
                    func(various_means), 
                    various_means_title, 
                    as_prob = True,
                    log_scale = log_scale,
                    save_to = normal_varying_path)
                
                
                uniform_mean1_path = os.path.join(save_to_folder, f'uniform_mean1_{dist.name}_dim{dim}_{func.__name__}.png')
                analysis.get_N_k_and_plot(
                    dist, 
                    k, 
                    func(uni_mean_1), 
                    uniform_mean1_title, 
                    as_prob = True,
                    log_scale = log_scale,
                    save_to = uniform_mean1_path)
        

# Consider data with F distribution in each of 3 and 20 dimensions

# dimensions = [3, 20]
# dist_functions = [ DistanceFunction.NORM_EUCLIDEAN ]

def make_f_distributed_data(dim: int, rng: np.random.Generator):
    f_dist_data = np.zeros((10000, dim))
    f_dist_mean_0 = np.zeros((10000, dim))
    f_dist_various_means = np.zeros((10000, dim))
    means = np.arange(-math.floor(dim/2), math.floor(dim/2) + 1)
    rng.shuffle(means)
    
    dfnum = 10
    dfden = 20
    f_dist_mean = dfden/(dfden - 2)
    
    for i in range(dim):        
        f_dist = rng.f(dfnum = dfnum, dfden = dfden, size = 10000)
        
        f_dist_data[:, i] = f_dist
        f_dist_mean_0[:, i] = f_dist - f_dist_mean            
        f_dist_various_means[:, i] = f_dist_mean_0[:, i] + means[i]
        
    return f_dist_data, f_dist_mean_0, f_dist_various_means



def plot_f_dist_N_k(
        k: int = 10,
        dimensions: List[int] = [3, 20, 768], 
        dist_functions: List[analysis.DistanceFunction] = [analysis.DistanceFunction.NORM_EUCLIDEAN],
        random_state: int = 0) -> None:
    rng = default_rng(random_state)

    for dim in dimensions:
        
        f_dist_data, f_dist_mean_0, f_dist_various_means = make_f_distributed_data(dim, rng)
        
        for dist in dist_functions:
            suffix = f', dim = {dim}, dist = {dist.value}'
            
            f_dist_title = f'F distribution dfnum = 10, dfden = 20{suffix}'
            f_dist_mean_0_title = f'F distribution mean 0, dfnum = 10, dfden = 20{suffix}'
            f_dist_various_title = f'F distribution various means, dfnum = 10, dfden = 20{suffix}'
            
            analysis.get_N_k_and_plot(
                dist, 
                k, 
                f_dist_data, 
                f_dist_title, 
                as_prob = True,
                log_scale = False,
                save_to = '')
            
            analysis.get_N_k_and_plot(
                dist, 
                k, 
                f_dist_mean_0, 
                f_dist_mean_0_title, 
                as_prob = True,
                log_scale = False,
                save_to = '')
            
            analysis.get_N_k_and_plot(
                dist, 
                k, 
                f_dist_various_means, 
                f_dist_various_title, 
                as_prob = True,
                log_scale = False,
                save_to = '')
            
            
            suffix = f', dim = {dim}, dist = {dist.value}, force_normal'
            
            f_dist_title = f'F distribution dfnum = 10, dfden = 20{suffix}'
            f_dist_mean_0_title = f'F distribution mean 0, dfnum = 10, dfden = 20{suffix}'
            f_dist_various_title = f'F distribution various means, dfnum = 10, dfden = 20{suffix}'
            
            f_dist_data_norm = reduction.transform_dimensions(
                f_dist_data,
                random_state,
                reduction.ProbabilityDistribution.NORMAL,
                normalize_rows = False)
            
            analysis.get_N_k_and_plot(
                dist, 
                k, 
                f_dist_data_norm, 
                f_dist_title, 
                as_prob = True,
                log_scale = False,
                save_to = '')
            
            
            f_dist_mean_0_norm = reduction.transform_dimensions(
                f_dist_mean_0,
                random_state,
                reduction.ProbabilityDistribution.NORMAL,
                normalize_rows = False)
            
            analysis.get_N_k_and_plot(
                dist, 
                k, 
                f_dist_mean_0_norm, 
                f_dist_mean_0_title, 
                as_prob = True,
                log_scale = False,
                save_to = '')
            
            
            f_dist_various_means_norm = reduction.transform_dimensions(
                f_dist_various_means,
                random_state,
                reduction.ProbabilityDistribution.NORMAL,
                normalize_rows = False)
            
            analysis.get_N_k_and_plot(
                dist, 
                k, 
                f_dist_various_means_norm,
                f_dist_various_title, 
                as_prob = True,
                log_scale = False,
                save_to = '')
            







