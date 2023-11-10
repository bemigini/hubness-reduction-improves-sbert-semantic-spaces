# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 12:02:37 2022

@author: bmgi


Center features


"""



import torch
from torch import Tensor
from torch import nn
from typing import Dict


class CenterEmbeddings(nn.Module):
    """
    This layer centers the embeddings. 
    That is, it gives every embedding a mean of 0.
    """
    def __init__(self):
        super(CenterEmbeddings, self).__init__()
        
        
    def forward(self, features: Dict[str, Tensor]):
        
        sentence_embeddings = features['sentence_embedding']
        embedding_means = torch.mean(sentence_embeddings, dim = 1)
        
        centered_embeddings = sentence_embeddings - embedding_means.reshape(-1, 1)
        
        features.update({'sentence_embedding': centered_embeddings})
        return features
    

    def save(self, output_path):
        pass
        
    
    @staticmethod
    def load(input_path):
        return CenterEmbeddings()


