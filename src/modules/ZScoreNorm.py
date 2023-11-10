# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:09:42 2022

@author: bmgi


Z score normalization module


"""


import torch
from torch import Tensor
from torch import nn
from typing import Dict


class ZScoreNorm(nn.Module):
    """
    This layer performs z score normalization on the embeddings.
    We use unbiased = False in the standard deviation. In Fei et al. 2022,
    they write their standard deviation without Bessel's correction, but also
    say that they scale their vectors to have length sqrt(D) where D is the 
    dimension of the vector. Using standard deviation without Bessel's 
    correction gives a vector length of sqrt(D + 1) where D is the 
    dimension of the vector. This does not make much difference if vectors
    have a large number of dimensions.
    """
    def __init__(self):
        super(ZScoreNorm, self).__init__()
        
        
    def forward(self, features: Dict[str, Tensor]):
        
        sentence_embeddings = features['sentence_embedding']        
        
        embedding_means = torch.mean(sentence_embeddings, dim = 1)
        embedding_std_dev = torch.std(sentence_embeddings, dim = 1, unbiased = False)
        # Avoid division by zero
        embedding_std_dev = embedding_std_dev + 1e-6
        
        z_normed = (sentence_embeddings - embedding_means.reshape(-1,1))/embedding_std_dev.reshape(-1,1)
        
        features.update({'sentence_embedding': z_normed})
        return features
    

    def save(self, output_path):
        pass
        
    
    @staticmethod
    def load(input_path):
        return ZScoreNorm()








