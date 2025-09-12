
from transformers import AutoTokenizer, AutoModel

from utils import load_dictionary, load_queries



dictionary_path = "./data/raw/train_dictionary.txt"
queries_dir = "./data/raw/traindev"
encoder_name = 'dmis-lab/biobert-base-cased-v1.1' #Dense encoder model nmae


dictionary = load_dictionary(dictionary_path)
queries  = load_queries(queries_dir)

tokenizer = AutoTokenizer.from_pretrained(encoder_name)
encoder = AutoModel.from_pretrained(encoder_name)







