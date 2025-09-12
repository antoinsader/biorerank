
import torch
import os
from transformers import AutoTokenizer
import time
import numpy as np

from utils import load_dictionary, load_queries,tokenize_names_with_memmap
import config as conf



print(f"init tokenizer from {conf.encoder_name}")
tokenizer = AutoTokenizer.from_pretrained(conf.encoder_name)


train_dictionary  = load_dictionary(conf.train_dictionary_path)
train_queries = load_queries(conf.train_queries_dir)

query_names, query_ids = [row[0] for row in train_queries], [row[1] for row in train_queries]
dictionary_names, dictionary_ids = [row[0] for row in train_dictionary], [row[1] for row in train_dictionary]



os.makedirs(conf.queries_dir, exist_ok=True)
os.makedirs(conf.dictionary_dir, exist_ok=True)



np.save(conf.queries_mmap_base + conf.ids_file_suffix, query_ids)
np.save(conf.dictionary_mmap_base + conf.ids_file_suffix, dictionary_ids)


t0 = time.time()
tok_queries =  tokenize_names_with_memmap(
    tokenizer = tokenizer, 
    names=query_names, 
    mmap_path_base=conf.queries_mmap_base, 
    max_length=conf.max_length, 
    tokenized_inputs_suffix=conf.tokens_inputs_file_suffix, 
    tokenized_att_suffix=conf.tokens_attentions_file_suffix
)
if tok_queries:
    print(f"query tokenized in dir: {conf.queries_dir}, took time: {time.time()-t0}s")

t0 = time.time()
tok_dicts = tokenize_names_with_memmap(
    tokenizer = tokenizer, 
    names=dictionary_names, 
    mmap_path_base=conf.dictionary_mmap_base, 
    max_length=conf.max_length, 
    tokenized_inputs_suffix=conf.tokens_inputs_file_suffix, 
    tokenized_att_suffix=conf.tokens_attentions_file_suffix
)
if tok_dicts:
    print(f"dicts tokenized in dir: {conf.dictionary_dir}, took time: {time.time()-t0}s")



