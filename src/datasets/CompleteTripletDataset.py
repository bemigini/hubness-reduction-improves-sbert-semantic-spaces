# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:56:47 2022

@author: bmgi


Triplet Dataset which yields all triplet combinations


"""


import logging

import math
import numpy as np
from numpy.typing import NDArray

from sentence_transformers.readers import InputExample

from torch.utils.data import  IterableDataset
from tqdm import tqdm
from typing import Dict, List, Union



logger = logging.getLogger(__name__)


class CompleteTripletDataset(IterableDataset):
    """
    For returning Lists of InputExample which are triplets.
    """
    def __init__(
            self, 
            examples: List[InputExample],
            make_all_combinations: bool = False,
            shuffle: bool = False,
            chunks: int = 10):
        """        
        :param examples:
            a list with InputExamples with a text and a label  
        :param make_all_combinations:
            make all possible combinations
        :param shuffle:
            whether to shuffle dataset
        """
        super(CompleteTripletDataset).__init__()

        self.texts_with_labels = np.array(examples)            
        self.make_all_combinations = make_all_combinations
        self.shuffle = shuffle
        self.through_all_triplets = False
        
        self.labels = np.unique([example.label for example in examples])
        
        label_to_indexes = {label: [] for label in self.labels}    
        for i, example in enumerate(examples):
            label_to_indexes[example.label].append(i)
        
        self.label_to_indexes = {label: np.array(label_to_indexes[label]) 
                                 for label in label_to_indexes}
        
        self.chunks = chunks
        self.current_chunk = 0
        self.chunk_triplets = np.array([])
        self.through_chunk = False
        
    
    def make_index_triplets(
            self,
            label_to_indexes: Dict[Union[int, float], List[int]],
            make_all_combinations: bool = False) -> NDArray:
        
                
        triplets = np.array([])
        
        for label in tqdm(label_to_indexes):
            current_label_indexes = label_to_indexes[label]
            
            current_label_grid = np.meshgrid(
                current_label_indexes, 
                current_label_indexes, 
                indexing = 'ij')
            uppertriangle_indexes = np.triu_indices(len(current_label_indexes), 1)
            first_idx = current_label_grid[0][uppertriangle_indexes]
            second_idx = current_label_grid[1][uppertriangle_indexes]   
                        
            other_labels = [key 
                            for key in label_to_indexes
                            if key != label]
            
            flat_different_labels = np.array([])
            
            for key in other_labels:
                flat_different_labels = np.concatenate(
                    (flat_different_labels, label_to_indexes[key]))            
            
            pos_length = len(first_idx)
            diff_length = len(flat_different_labels)
            
            if make_all_combinations:
                current_triplets = np.zeros((pos_length*diff_length, 3))
                
                positive_matrix = np.array((first_idx, second_idx))
                
                pos_with_repeats = np.repeat(
                    positive_matrix, 
                    diff_length, 
                    axis = 1)
                diff_repeats = np.tile(
                    flat_different_labels,
                    pos_length)
                
                current_triplets[:, 0] = pos_with_repeats[0]
                current_triplets[:, 1] = pos_with_repeats[1]
                current_triplets[:, 2] = diff_repeats                
                
            else:
                # For each anchor have at least one triplet with each positive
                # example and at least one triplet with each negative example,
                # but there will not be each combination of positive examples
                # paired with each negative.                
                longest = max(pos_length, diff_length)
                current_triplets = np.zeros((longest, 3))
                
                if longest > pos_length:
                    first_idx_repeats = np.tile(
                        first_idx,
                        math.ceil(longest / pos_length))
                    second_idx_repeats = np.tile(
                        second_idx,
                        math.ceil(longest / pos_length))                    
                    
                    diff_repeats = flat_different_labels
                else:
                    first_idx_repeats = first_idx
                    second_idx_repeats = second_idx
                    diff_repeats = np.tile(
                        flat_different_labels,
                        math.ceil(longest / diff_length))
                
                current_triplets[:, 0] = first_idx_repeats[0:longest]
                current_triplets[:, 1] = second_idx_repeats[0:longest]
                current_triplets[:, 2] = diff_repeats[0:longest]
                
            if triplets.shape[0] == 0:
                triplets = current_triplets
            else:
                triplets = np.concatenate((triplets, current_triplets), axis = 0)
                
            del current_triplets                
            
        return triplets.astype(int)
        
    
    def init_dataset_chunk(self, chunk_number: int):
        
        chunk_labels_to_indexes = {}
        
        for label in self.label_to_indexes:
            all_indexes = self.label_to_indexes[label]
            len_indexes = len(all_indexes)
            chunk_size = math.ceil(len_indexes / self.chunks)
            chunk_indexes = all_indexes[self.current_chunk * chunk_size : (self.current_chunk + 1) * chunk_size]
            
            chunk_labels_to_indexes[label] = chunk_indexes
               
        
        self.chunk_triplets = self.make_index_triplets(
            label_to_indexes = chunk_labels_to_indexes,
            make_all_combinations = self.make_all_combinations) 
        
        if self.shuffle:            
            np.random.shuffle(self.chunk_triplets)
       

    def __iter__(self):
        if self.through_all_triplets:
            self.current_chunk = 0
            self.chunk_triplets = np.array([])
            self.through_chunk = False
            self.through_all_triplets = False
            
            if self.shuffle:
                for label in self.label_to_indexes:
                    np.random.shuffle(self.label_to_indexes[label])
            
            
        if self.through_chunk == True:
            self.current_chunk += 1
            self.chunk_triplets = np.array([])
            self.through_chunk = False                        
        
        if not self.chunk_triplets.any():
            self.init_dataset_chunk(self.current_chunk)
            
        
        for i in range(len(self.chunk_triplets)):
            if (i + 1) == len(self.chunk_triplets):
                self.through_chunk = True
                if (self.current_chunk + 1) == self.chunks:
                    self.through_all_triplets = True                    
            
            triplet_indexes = self.chunk_triplets[i]
            
            triplet_input_examples = self.texts_with_labels[triplet_indexes]
            
            yield InputExample(
                texts = [
                    triplet_input_examples[0].texts[0],
                    triplet_input_examples[1].texts[0],
                    triplet_input_examples[2].texts[0]])
        

    def __len__(self):
        if not self.chunk_triplets.any():
            self.init_dataset_chunk(self.current_chunk)
        return len(self.chunk_triplets) * self.chunks







