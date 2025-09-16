
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

    os.makedirs("./data/draft", exist_ok=True)

    metrics = MetricsLogger(logger=LOGGER, confs=config)

    t0 = time.time()
    ds = MyDataSet(confs=config)

    LOGGER.info(f"dictionary_cuis: {len(ds.dictionary_cuis)} cuis, queries_cuis: ${len(ds.queries_cuis)} cuis")

    data_loader = DataLoader(
        ds,
        batch_size=config.train_batch_size,
        shuffle=True
    )
    model = Reranker(config)
    model.load_encoder(config.encoder_name)
    metrics.log_event("load_classes", t0)

    t0 = time.time()
    faiss_index = FaissIndex(config=config, encoder=model.encoder, tokens=ds.tokens)
    metrics.log_event("faiss_index_init", t0)

    result_encoder_dir = config.result_encoders_dir + f"/encoder_{current_global_log_number}/" 
    os.makedirs(result_encoder_dir, exist_ok=True)


    start = time.time()

    best_acc_1, acc_5, best_epoch = 0.0,.0,0


    for epoch in  tqdm(range(1,config.num_epochs+1)):   
        t0 = time.time()
        faiss_index.build_index()
        metrics.log_event(f"epoch_{epoch}: faiss_index_buid", t0)
        t0 = time.time()
        dictionary_candidates_idxs = faiss_index.search_candidates()
        metrics.log_event(f"epoch_{epoch}: faiss_index_search", t0)

        ds.set_dense_candidate_idxs(dictionary_candidates_idxs)

        train_loss = 0.0
        correct_att1, correct_att5, total = 0, 0, 0
        model.train()
        t0=time.time()
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader), unit="batch", desc="Training batches"):
            model.optimizer.zero_grad()
            batch_x, batch_y = data
            queries_tokens, candidates_tokens = batch_x
            scores = model(queries_tokens, candidates_tokens)
            loss = model.get_loss(scores, batch_y)
            loss.backward()
            model.optimizer.step()
            train_loss += loss.item()

            with torch.no_grad():
                # batch_y shape: (batch_size, topk)
                batch_y_true_indices = batch_y.argmax(dim=1).to(scores.device) # (batch_size) #for each query what is the index of label 1
                #acc@1
                #scores  shape: (batch_size, topk)
                pred_true_indices = scores.argmax(1) #(batch_size) #for each query what is the index where scores are the maximum
                correct_att1 += (pred_true_indices == batch_y_true_indices).sum().item()

                #acc@5
                pred_true_indices_top5 = scores.topk(5, dim=1).indices #(batch_size, 5)
                correct_att5 += (pred_true_indices_top5 == batch_y_true_indices.unsqueeze(1)).any(dim=1).sum().item()


                total += batch_y.shape[0]

        metrics.log_event(f"epoch_{epoch}: batches loop finished", t0 )
        accuracy_1 = correct_att1 / total
        accuracy_5 = correct_att5 / total

        if accuracy_1 > best_acc_1:
            best_acc_1 = accuracy_1
            best_epoch = epoch
            acc_5 = accuracy_5
            model.save_state(result_encoder_dir)




        LOGGER.info(f"Epoch {epoch} accuracy: acc@1: {accuracy_1:.4f}, acc@5: {accuracy_5:.4f}")



    metrics.log_event(f"finished training", t0=start)

    training_time = time.time()-start
    training_time_str = f"{int(training_time/60/60)}h, {int(training_time/60 % 60)}mins, {int(training_time % 60)}secs"


    #Log in global:
    with open(config.global_log_path, "w") as f:
        log_global_data[-1]["queries size"]  = len(ds.queries_cuis)
        log_global_data[-1]["dictionary size"]  = len(ds.dictionary_cuis)
        log_global_data[-1]["finished time"]  = training_time_str
        log_global_data[-1]["log details file"]  = log_path
        log_global_data[-1]["encoder dir"]  = result_encoder_dir
        log_global_data[-1]["encoder epoch"]  = best_epoch
        log_global_data[-1]["acc@1"]  = best_acc_1
        log_global_data[-1]["acc@5"]  = acc_5
        json.dump(log_global_data,f)

    LOGGER.info(f"Training Time: {training_time_str} ")
    LOGGER.info(f"Logs saved in: {log_path}")




if __name__=="__main__":
    main()

