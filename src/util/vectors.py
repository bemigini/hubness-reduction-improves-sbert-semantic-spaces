# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:03:32 2022

@author: bmgi


vector utilities


"""


from enum import Enum

import numpy as np
from numpy.typing import NDArray



class VectorRelation(Enum):
    ORTHOGONAL = 0
    OPPOSITE = 1


def vector_dist(v1: NDArray, v2: NDArray) -> float:
    return np.linalg.norm(v1 - v2, ord = 2)


def normalize_vector(v: NDArray) -> NDArray:
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm


def normalized_vector_dist(v1: NDArray, v2: NDArray) -> float:
    normed_v1 = normalize_vector(v1)
    normed_v2 = normalize_vector(v2)    
    
    return vector_dist(normed_v1, normed_v2)


