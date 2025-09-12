from tqdm import tqdm
import glob 
import os
import numpy as np
import json
import logging
import json
from datetime import datetime
import pickle

def init_logging(confs, logger):
    global_log_path = confs.global_log_path

    log_global_data = []

    if not os.path.isfile(global_log_path):
        with open(global_log_path, "w") as f:
            json.dump(log_global_data,f)

    with open(global_log_path, "r") as f:
        log_global_data = json.load(f)


    last_log_number  = log_global_data[-1]["id"] if len(log_global_data) > 0  else 0
    current_global_log_number = last_log_number + 1 
    log_global_data.append({"id": current_global_log_number})

    with open(global_log_path, "w") as f:
        json.dump(log_global_data, f)


    os.makedirs(confs.logs_dir, exist_ok=True)
    datestr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = confs.logs_dir + f"/log_{current_global_log_number}_{datestr}.log"
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(message)s')

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)


    return log_path,  log_global_data,current_global_log_number



def load_dictionary(dict_path):
    data = []
    with open(dict_path, mode="r", encoding="utf-8") as f:
        lines =  f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            if line == "": continue
            cui, name  = line.split("||")
            data.append((name, cui))
        data = np.array(data)
        return data

def load_queries(data_dir):
    data = []
    concept_files = glob.glob(os.path.join(data_dir, "*.concept"))

    for concept_file in tqdm(concept_files):
        with open(concept_file, "r", encoding='utf-8') as f:
                concepts = f.readlines()
        for concept in concepts:
            concept = concept.split("||")
            mention = concept[3].strip()
            cui = concept[4].strip()
            is_composite = (cui.replace("+","|").count("|") > 0)
            if is_composite:
                continue
            if cui == "-1":
                continue
            data.append((mention, cui))
        data = list(dict.fromkeys(data))

    data = np.array(data)
    return data


def tokenize_names_with_memmap(
    tokenizer, 
    names, 
    mmap_path_base, 
    max_length, 
    tokenized_inputs_suffix, 
    tokenized_att_suffix, 
    batch_size = 4096):

    if isinstance(names, np.ndarray):
        names  = names.tolist()
    names_size = len(names)

    #create mmap conf for input_ids and attention mask
    input_ids_mmap_path = mmap_path_base + tokenized_inputs_suffix
    attention_mask_mmap_path = mmap_path_base + tokenized_att_suffix


    input_ids_array = np.memmap(input_ids_mmap_path,
                                mode="w+",
                                dtype=np.int32,
                                shape=(names_size, max_length)
                                )
    att_mask_array = np.memmap(attention_mask_mmap_path,
                                mode="w+",
                                dtype=np.int32,
                                shape=(names_size, max_length)
                                )
    #saving meta 
    _meta = {"shape": (names_size, max_length)}
    with open(mmap_path_base + ".json", "w") as f:
        json.dump(_meta, f)

    #tokenize in epochs
    for start in tqdm(range(0, names_size, batch_size), desc="tokenizing", unit="batch"):
        end = min(start + batch_size , names_size)
        names_batch = names[start: end]
        tokens = tokenizer(
            names_batch,
            padding="max_length", 
            max_length=max_length, 
            truncation=True, 
            return_tensors="pt"
        )
        input_ids_array[start:end, : ] = tokens["input_ids"]
        att_mask_array[start:end, : ] = tokens["attention_mask"]
    input_ids_array.flush()
    att_mask_array.flush()
    return True



def save_pkl(ar, fp):
    with open(fp, 'wb') as f:
        pickle.dump(ar, f)
def get_pkl(fp):
    with open(fp, "rb") as f:
        return pickle.load(f)

