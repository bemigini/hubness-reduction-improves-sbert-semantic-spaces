# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:01:40 2022

@author: bmgi


Finding normalized euclidean distance pytorch implementation


"""


import numpy as np
from numpy.typing import NDArray

import torch 



def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(embeddings, p=2, dim=1)
    mask = (norms != 0)
    normalized_emb = embeddings
    normalized_emb[mask] = normalized_emb[mask].div(norms[mask])
    
    return normalized_emb


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









