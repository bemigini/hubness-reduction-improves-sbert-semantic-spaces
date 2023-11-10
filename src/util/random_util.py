# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 09:04:39 2022

@author: bmgi


random util


"""


import numpy as np

import torch


def set_seed(seed: int, cuda: bool) -> None:
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)


