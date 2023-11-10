# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:49:43 2022

@author: bmgi


Triplet Dataset which yields random samples of triplets


"""



from torch.utils.data import  IterableDataset
import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Union
from sentence_transformers.readers import InputExample
import logging

logger = logging.getLogger(__name__)


class SampleTripletDataset(IterableDataset):
    """
    For returning Lists of InputExample which are triplets.
    """
    def __init__(
            self, 
            examples: List[InputExample], 
            triplets_per_anchor: int = 2, 
            with_replacement: bool = False):
        """        
        :param examples:
            a list with InputExamples with a text and a label
        :param triplets_per_anchor:
            how many triplets to construct for each anchor text
        :param with_replacement:
            if this is True, then the same positive/negative example for an anchor can be drawn multiple times.
            if this is False, each positive/negative example is drawn at most once.
            drawing.
        """
        super(SampleTripletDataset).__init__()
        
        self.texts_with_labels = np.array(examples)
        self.triplets_of_indexes_for_texts = np.array([])
        self.with_replacement = with_replacement
        self.reinit = False
        
        self.triplets_per_anchor = triplets_per_anchor        
        self.labels = np.unique([example.label for example in examples])
        
        label_to_indexes = {label: [] for label in self.labels}    
        for i, example in enumerate(examples):
            label_to_indexes[example.label].append(i)
            
        self.label_to_indexes = label_to_indexes
        
        
    def get_sample_size(
            self, 
            population: List, 
            wanted_sample_size: int, 
            with_replacement: bool) -> int:
        if with_replacement:
            sample_size = wanted_sample_size                
        else:
            sample_size = min(wanted_sample_size, len(population))
            
        return sample_size
    
    
    def make_index_triplets(
            self,
            input_examples: List[InputExample], 
            label_to_indexes: Dict[Union[int, float], List[int]],
            triplets_per_anchor: int = 1,
            with_replacement: bool = False) -> NDArray:
        
        triplets = []        
        
        
        for i, example in enumerate(input_examples):
            current_index = i
            current_label = example.label
            
            same_label_without_current = [index 
                                          for index 
                                          in label_to_indexes[current_label] 
                                          if index != current_index]
            
            sample_size = self.get_sample_size(
                same_label_without_current, 
                triplets_per_anchor, 
                with_replacement)            
            
            same_label_indexes = np.random.choice(
                same_label_without_current, 
                size = sample_size,
                replace = with_replacement)
            
            same_idx_size = len(same_label_indexes)
            
            
            different_kinds = [label_to_indexes[key] 
                                      for key in label_to_indexes 
                                      if key != current_label]
            flat_different_kinds = [index 
                                    for sublist in different_kinds 
                                    for index in sublist]
            
            sample_size = self.get_sample_size(
                flat_different_kinds, 
                triplets_per_anchor, 
                with_replacement)    
            
            different_indexes = np.random.choice(
                flat_different_kinds, 
                size = sample_size,
                replace = with_replacement)
            
            different_idx_size = len(different_indexes)
            
            for trip in range(triplets_per_anchor):
                triplets.append(np.array([
                    current_index, 
                    same_label_indexes[trip % same_idx_size], 
                    different_indexes[trip % different_idx_size]]))
            
        return np.array(triplets)
    
    
    def make_input_example_triplets(
            self,
            input_examples: List[InputExample], 
            label_to_indexes: Dict[Union[int, float], List[int]],
            triplets_per_anchor: int = 1,
            with_replacement: bool = False) -> NDArray:
        
        triplets_indexes = self.make_index_triplets(
            input_examples,
            label_to_indexes,
            triplets_per_anchor,
            with_replacement)
        
        input_examples_np = np.array(input_examples)
        
        triplet_input_examples = [
            InputExample(
                texts = input_examples_np[triplets_indexes[i]]) 
            for i in range(len(triplets_indexes))]        
            
        return triplet_input_examples
    
    
    def init_dataset(self):
        self.triplets_of_indexes_for_texts = self.make_index_triplets(
            input_examples = self.texts_with_labels,
            label_to_indexes = self.label_to_indexes,
            triplets_per_anchor = self.triplets_per_anchor,
            with_replacement = self.with_replacement)    
        
    
    def shuffle_dataset(self) -> None:
        np.random.shuffle(self.triplets_of_indexes_for_texts)
        
        
    def build_dataset(self):
        del self.triplets_of_indexes_for_texts
        self.init_dataset()
        self.shuffle_dataset()
        self.reinit = False
        
    
    def __iter__(self):
        if (not self.triplets_of_indexes_for_texts.any()) or self.reinit:
            self.build_dataset()
            
        for i in range(len(self.triplets_of_indexes_for_texts)):
            if i + 1 == len(self.triplets_of_indexes_for_texts):
                self.reinit = True
            
            triplet_indexes = self.triplets_of_indexes_for_texts[i]
            
            triplet_input_examples = self.texts_with_labels[triplet_indexes]
            
            yield InputExample(
                texts = [
                    triplet_input_examples[0].texts[0],
                    triplet_input_examples[1].texts[0],
                    triplet_input_examples[2].texts[0]])
        

    def __len__(self):
        return len(self.texts_with_labels) * self.triplets_per_anchor





