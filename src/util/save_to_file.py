# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:53:15 2022

@author: bemigini

For writing objects to files


"""


from enum import Enum
import h5py

import json
from json import JSONEncoder

import numpy as np
from numpy.typing import NDArray

import os
import matplotlib.pyplot as plt

from tqdm import tqdm


class CustomEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Enum):
            return obj.name
        if obj is None:
            return ''          
        return obj.__dict__


def save_as_json(obj: object, save_to: str) -> None:
    json_str = json.dumps(obj, indent = 4,
                          cls = CustomEncoder)
    
    with open(save_to, 'w') as f:
        f.write(json_str)


def load_json(load_from: str) -> str:
    
    with open(load_from, 'r') as f:
        json_obj = json.load(f)
        
    return json_obj


def save_to_hdf5(data: NDArray, h5_file_path: str, h5_dataset_name: str) -> None:
    with h5py.File(h5_file_path, 'w') as h5f:
        h5f.create_dataset(h5_dataset_name, data=data, compression='gzip')


def convert_csv_gz_to_hdf5(file_path: str, dataset_name: str) -> None:
    folder_name = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    h5_file_name = file_name.replace('.csv.gz', '.h5')
    h5_file_path = os.path.join(folder_name, h5_file_name)
    
    if not os.path.exists(h5_file_path):
        embeddings = np.loadtxt(file_path, delimiter = ',')
        save_to_hdf5(embeddings, h5_file_path, dataset_name)
    

def load_from_hdf5(h5_file_path: str, dataset_name: str) -> NDArray:
    with h5py.File(h5_file_path, 'r') as h5f:
        data = h5f[dataset_name][:]
    
    return data


def convert_all_csv_gz_in_folder_to_hdf5(
        folder_path: str, dataset_name: str) -> None:
    
    csv_gz_files = [file 
                    for file in os.listdir(folder_path) 
                    if file.endswith('.csv.gz')]
    
    for file in tqdm(csv_gz_files):
        convert_csv_gz_to_hdf5(os.path.join(folder_path, file), dataset_name)
    
    
def save_pdf_fig(file_name: str, save_folder: str):
    save_name = f'{file_name}.pdf'
    save_path = os.path.join(save_folder, save_name)
    
    plt.savefig(save_path, bbox_inches='tight')


def save_high_res_png(file_name: str, save_folder: str):
    save_name = f'{file_name}.png'
    save_path = os.path.join(save_folder, save_name)
    
    plt.savefig(save_path, dpi = 600, bbox_inches='tight')
    
    
    



