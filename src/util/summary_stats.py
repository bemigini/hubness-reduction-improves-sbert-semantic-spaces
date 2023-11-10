# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 08:42:52 2022

@author: bmgi


A class for summary statistics


"""


import statistics 
from typing import List


class SummaryStats():
    def __init__(self, values: List[float]):
        
        self.max_value = max(values)
        self.min_value = min(values)
        self.mean = statistics.mean(values)
        self.standard_deviation = statistics.stdev(values)





