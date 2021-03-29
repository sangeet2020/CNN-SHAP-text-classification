#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   2021-02-23 01:52:54
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlandes
# @Last Modified time: 2021-03-18 01:32:15

"""
Load pre-trained embeddings and create embedding weight matrix
"""

import os
import sys
import argparse
import numpy as np
import torch.nn as nn
import torch
from gensim.models import KeyedVectors
from arguments import parse_arguments

args = parse_arguments()


class MyEmbedding(object):
    def __init__(self, params, path=args.emb_f):
        self.path = path
        self.embedding_size = params.embedding_size
    
    def load_embeddings(self, t_words):
        print("\nLoading pre-trained embeddings...")
        if not os.path.exists(self.path):
            print("Embeddings not found")
        self.word2vec = KeyedVectors.load_word2vec_format(self.path)
        len_words = len(t_words.word_index) + 1
        self.embedding_weights = np.zeros((len_words, self.embedding_size))
        word2id = t_words.word_index
        for word, index in word2id.items():
            try:
                self.embedding_weights[index, :] = self.word2vec[word]
            except KeyError:
                pass
        
        print("--Done--")
        return self.embedding_weights
        

        


