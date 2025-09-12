import torch

# [GENERAL]
train_dictionary_path = "./data/data-ncbi-fair/train_dictionary.txt"
train_queries_dir= "./data/data-ncbi-fair/traindev/"
output_dir = "./data/output"
encoder_name = "dmis-lab/biobert-base-cased-v1.1"
global_log_path = "./data/global_log.json"
logs_dir = "./data/logs"

use_cuda = torch.cuda.is_available()
use_fp16 = True
amp_dtype = torch.bfloat16


# [tokenizer]
max_length= 25
tokenizer_output_dir = "./data/output/tokenized"
queries_dir, dictionary_dir = tokenizer_output_dir+  '/queries', tokenizer_output_dir+ '/dictionary'
queries_files_prefix, dictionary_files_prefix = "/queries_", "/dictionary_"
queries_mmap_base, dictionary_mmap_base = queries_dir + queries_files_prefix, dictionary_dir + dictionary_files_prefix
ids_file_suffix,tokens_inputs_file_suffix, tokens_attentions_file_suffix = '_ids.npy',  '_inputids.mmap', '_attentionmask.mmap'



# [training]
result_encoder_dir = output_dir + "/encoder"
num_epochs  = 10
topk = 20
faiss_build_batch_size = 4096
faiss_search_batch_size = 4096
train_batch_size = 68
weight_decay = 0.01 
learning_rate = 1e-5

