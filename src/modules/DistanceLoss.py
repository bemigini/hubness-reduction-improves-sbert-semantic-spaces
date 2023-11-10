# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 12:38:43 2022

@author: bmgi


DistanceLoss, based on CosineSimilarityLoss from sentence_transformers

https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/CosineSimilarityLoss.py



"""


from enum import Enum

from sentence_transformers import SentenceTransformer

from torch import nn, Tensor
import torch.nn.functional as F
from typing import Iterable, Dict


class DistanceFunction(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)


class DistanceLoss(nn.Module):
    """
    DistanceLoss expects, that the InputExamples consists of two texts and a float label.
    It computes the vectors u = model(input_text[0]) and v = model(input_text[1]) and measures the distance between the two.
    By default, it minimizes the following loss: ||input_label - dist_transformation(dist(u,v))||_2.
    :param model: SentenceTranformer model
    :param loss_fct: Which pytorch loss function should be used to compare the dist(u,v) with the input_label? By default, MSE:  ||input_label - dist(u,v)||_2
    :param dist_transformation: The dist_transformation function is applied on top of dist. By default, the identity function is used (i.e. no change).
    Example::
            from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample
            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.2),
                InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.7)]
            train_dataset = SentencesDataset(train_examples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = DistanceLoss(model=model)
    """
    def __init__(
            self, 
            model: SentenceTransformer, 
            dist = DistanceFunction.EUCLIDEAN, 
            loss_fct = nn.MSELoss(), 
            dist_transformation=nn.Identity()):
        super(DistanceLoss, self).__init__()
        self.model = model
        self.dist = dist
        self.loss_fct = loss_fct
        self.dist_transformation = dist_transformation


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        output = self.dist_transformation(self.dist(embeddings[0], embeddings[1]))
        return self.loss_fct(output, labels.view(-1))







