
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import config
from classes import FaissIndex, MetricsLogger, Reranker, MyDataSet
import time
import logging
import json
from datetime import datetime

from utils import init_logging








LOGGER = logging.getLogger()




def main():
    log_path,  log_global_data, current_global_log_number= init_logging(confs=config, logger=LOGGER)

    metrics = MetricsLogger(logger=LOGGER, confs=config)

    t0 = time.time()
    ds = MyDataSet(confs=config)
    data_loader = DataLoader(
        ds,
        batch_size=config.train_batch_size,
        shuffle=True
    )
    model = Reranker(config)
    metrics.log_event("load_classes", t0)

    t0 = time.time()
    faiss_index = FaissIndex(config=config, encoder=model.encoder, tokens=ds.tokens)
    metrics.log_event("faiss_index_init", t0)

    result_encoder_dir = config.result_encoders_dir + f"/encoder_{current_global_log_number}/" 
    os.makedirs(result_encoder_dir, exist_ok=True)


    start = time.time()

    for epoch in  tqdm(range(1,config.num_epochs+1)):   
        t0 = time.time()
        faiss_index.build_index()
        metrics.log_event(f"epoch_{epoch}: faiss_index_buid", t0)
        t0 = time.time()
        dictionary_candidates_idxs = faiss_index.search()
        metrics.log_event(f"epoch_{epoch}: faiss_index_search", t0)

        ds.set_dense_candidate_idxs(dictionary_candidates_idxs)

        train_loss = 0.0
        model.train()
        t0=time.time()
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader), unit="batch", desc="Training batches"):
            model.optimizer.zero_grad()
            batch_x, batch_y = data
            scores = model(batch_x)
            loss = model.get_loss(scores, batch_y)
            loss.backward()
            model.optimizer.step()
            train_loss += loss.item()
        metrics.log_event(f"epoch_{epoch}: batches loop finished", t0)


        if epoch == config.num_epochs:
            model.save_state(result_encoder_dir)

    metrics.log_event(f"finished training", t0=start)

    training_time = time.time()-start
    training_time_str = f"{int(training_time/60/60)}h, {int(training_time/60 % 60)}mins, {int(training_time % 60)}secs"


    #Log in global:
    with open(config.global_log_path, "w") as f:
        log_global_data[-1]["queries size"]  = len(ds.query_ids)
        log_global_data[-1]["dictionary size"]  = len(ds.dictionary_ids)
        log_global_data[-1]["finished time"]  = training_time_str
        log_global_data[-1]["log details file"]  = log_path
        log_global_data[-1]["encoder dir"]  = result_encoder_dir
        json.dump(log_global_data,f)

    LOGGER.info(f"Training Time: {training_time_str} ")
    LOGGER.info(f"Logs saved in: {log_path}")




if __name__=="__main__":
    main()

