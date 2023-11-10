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


class CenterFeatures(nn.Module):
    """
    This layer performs z score normalization on the feature vectors of the embeddings
    """
    def __init__(self):
        super(CenterFeatures, self).__init__()
        
        
    def forward(self, features: Dict[str, Tensor]):
        
        sentence_features = features['sentence_embedding']
        feature_means = torch.mean(sentence_features, dim = 0)
        
        centered_features = sentence_features - feature_means
        
        features.update({'sentence_embedding': centered_features})
        return features
    

    def save(self, output_path):
        pass
        
    
    @staticmethod
    def load(input_path):
        return CenterFeatures()


