# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 08:04:55 2023

@author: bmgi

An exploration of the large hubs left by MP in 20 Newsgroups 


"""


import numpy as np
import os


from src.datasets import loading_datasets as loadd

from src.util import save_to_file #, naming, summary_stats, vectors



result_folder_path = 'output/r_20_st'
results_name = 'sts_bert_distilroberta-base_euclidean_ORTHOGONAL_z_False_n_True_c_False_seed1_newsgroups_hubness_results.json'


result_path = os.path.join(result_folder_path, results_name)

hub_json_dict = save_to_file.load_json(result_path)

reductions = ['mutual_proximity']

k_occurrences = [np.array(hub_json_dict[r]['train']['k_occurrence'])
                 for r in reductions]

k_occur = k_occurrences[0]

type(k_occur)
print(k_occur.shape)
high_k_occurrence_idx = np.where(k_occur > 100)[0]
print(k_occur[k_occur > 100])
# These have slightly different k-occurrences, 
# probably because we are using approximate MP
# [279 282 282 285 258 258 257 259 277 258]

newsgroups_train_no_meta = loadd.get_20_newsgoups_data('train')
sentences_train = newsgroups_train_no_meta.data

print(type(sentences_train))


high_k_occurrence_sentences = [sentences_train[i] 
                               for i in high_k_occurrence_idx]

print(len(high_k_occurrence_sentences)) # 10
print(high_k_occurrence_sentences)
# These are all whitespace
# k- occurences from above
# [279 282 282 285 258 258 257 259 277 258]
# [' ', '', '', '', '\n', '\n\n', '', '', '', '']




reductions_base = ['base']

k_occurrences_base = [np.array(hub_json_dict[r]['train']['k_occurrence'])
                 for r in reductions_base]

k_occur_base = k_occurrences_base[0]

type(k_occur_base)
print(k_occur_base.shape)
high_k_occurrence_idx_base = np.where(k_occur_base > 100)[0]
print(k_occur_base[k_occur_base > 100])
# All the whitespaces are at 256 and the message with much whitespace is 121
# array([256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 121])


high_k_occurrence_base_sentences = [sentences_train[i] 
                                    for i in high_k_occurrence_idx_base]

print(len(high_k_occurrence_base_sentences)) # 11
print(high_k_occurrence_base_sentences)
# These are mostly whitespace and one with a lot of whitespace in it
# [' ', '\n', '\n', '', '', '', '', '', '\n', '', 
# '\n  For a commerical package try WAVE from  Precision Visuals\n                                           505-530-6563\n\n  For a free package try KHOROS from University of New Mexico\n                                      508-277-6563\n                                   ftp from\n                              ptrg.eece.unm.edu\n\n    Login in anonyomus or ftp  with a valid email address as the password\n               cd /pub/khoros/release\n\n   That will get you to the right place.']





whitespace_or_empty = [sentence 
                       for sentence in sentences_train
                       if sentence == '' or sentence == '\n' or sentence == '\n\n']

print(len(whitespace_or_empty)) # 268
# So all the large hubs after MP are empty or whitespace, but not all the 
# empty or whitespace messages become large >100 hubs


sentences_train_np = np.array(sentences_train)

white_empty_idx = np.where(np.isin(sentences_train_np, ['', ' ', '\n', '\n\n']))[0]

print(white_empty_idx.shape) # (276,)
white_empty_k_occur_base = k_occur_base[white_empty_idx]
print(np.unique(white_empty_k_occur_base, return_counts = True))


white_empty_k_occur_MP = k_occur[white_empty_idx]
print(np.unique(white_empty_k_occur_MP, return_counts = True))









reductions_fn_mp = ['normal_all_mutual_proximity']

k_occurrences_fn_mp = [np.array(hub_json_dict[r]['train']['k_occurrence'])
                 for r in reductions_fn_mp]

k_occur_fn_mp = k_occurrences_fn_mp[0]
white_empty_k_occur_fn_mp = k_occur_fn_mp[white_empty_idx]
print(np.unique(white_empty_k_occur_fn_mp, return_counts = True))
# So in this case, f-norm makes it possible to tell the whitespace messages 
# apart, and we no longer have messages which are “disconnected” with respect 
# to the neighbourhood relation.























