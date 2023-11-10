# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 08:36:22 2022

@author: bemigini 


Small utility functions


"""


import string

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import nltk.tokenize as nltk_tok



def remove_duplicates_preserve_order(sequence):
    seen = set()
    seen_add = seen.add
    return [elm for elm in sequence if elm not in seen and not seen_add(elm)]


def tokenize_rm_non_wordnet(
        text, remove_punct = True, remove_stop = False, use_morphy = False):
    tokens = tokenize(text, remove_punct, remove_stop, use_morphy)
    return remove_non_wordnet_words(tokens)


def tokenize(text, remove_punct = True, remove_stop = False, use_morphy = False):
    
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    try: 
        text_tokens = nltk_tok.word_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        text_tokens = nltk_tok.word_tokenize(text)
        
    return_tokens = text_tokens
    
    
    if remove_stop:
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')        
            stop_words = set(stopwords.words('english'))  
        
        text_no_stop = [w for w in text_tokens if w.lower() not in stop_words]
        return_tokens = text_no_stop    
    
    
    if use_morphy:
        return_tokens = [wn.morphy(w) for w in return_tokens]
    
    return return_tokens


def remove_non_wordnet_words(tokens):
    return [token for token in tokens if wn.synsets(token)]


def remove_stopwords(text):
    text_tokens = nltk_tok.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    text_no_stop = [w for w in text_tokens if w.lower() not in stop_words]
    text = ' '.join(text_no_stop)
    return text
    
    
    



