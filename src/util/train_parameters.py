# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 08:51:38 2022

@author: bmgi


Training parameters


"""


from dataclasses import dataclass

from src.util import save_to_file, vectors

from typing import Dict, List


@dataclass
class CenteringParameters:
    normalize_embeddings: bool
    center_embeddings: bool
    z_score_norm: bool


@dataclass
class TrainParameters:    
    centering_parameters: List[CenteringParameters]
    base_models: List[str]
    distances: List[str]
    max_difference: List[vectors.VectorRelation]
    

def decode_train_parameters(d: Dict):
    
    center_dicts = d['centering_parameters']
    centers = []
    
    for center_dict in center_dicts:
        centers.append(CenteringParameters(
            center_dict['normalize_embeddings'],
            center_dict['center_embeddings'],
            center_dict['z_score_norm']))
    
    vector_relations = d['max_difference']
    vector_rel_enums = []
    for vector_rel in vector_relations:
        vector_rel_enums.append(vectors.VectorRelation[vector_rel])
    
    train_para = TrainParameters(centers,
                                 d['base_models'],
                                 d['distances'],
                                 vector_rel_enums)
    
    return train_para


def save_train_parameters(train_parameters: TrainParameters, save_to: str) -> None:
    save_to_file.save_as_json(train_parameters, save_to)


def load_train_parameters(load_from: str) -> TrainParameters:    
    json_dict = save_to_file.load_json(load_from)    
    train_parameters = decode_train_parameters(json_dict)
    
    return train_parameters
    
    




