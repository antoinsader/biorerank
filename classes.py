import os
import json
import time


from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os, time, json, psutil

from transformers import AutoModel

import faiss
import faiss.contrib.torch_utils
from utils import save_pkl


class Reranker(nn.Module):
    def __init__(self, config):
        super(Reranker, self).__init__()
        encoder = AutoModel.from_pretrained(config.encoder_name, use_safetensors=True,)
        if config.use_cuda:
            encoder = encoder.to("cuda")
            encoder = torch.compile(encoder)
        self.encoder =encoder
        self.use_cuda = config.use_cuda
        self.device = "cuda" if config.use_cuda else "cpu"
        self.hidden_size = getattr(getattr(encoder, "config", None), "hidden_size", None)
        self.amp_dtype = config.amp_dtype

        self.optimizer = optim.AdamW(
            self.encoder.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=config.use_cuda
        )
        self.criterion = self.marginal_nll


        self.max_length = config.max_length
        self.topk = config.topk
        self.use_cuda = config.use_cuda

    def forward(self, queries_tokens, candidates_tokens):
        batch_size, topk, max_length = candidates_tokens['input_ids'].shape
        assert topk == self.topk and max_length == self.max_length

        if self.use_cuda:
            queries_tokens["input_ids"] = queries_tokens["input_ids"].to(self.device, non_blocking=True)
            queries_tokens["attention_mask"] = queries_tokens["attention_mask"].to(self.device, non_blocking=True)

        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_cuda):
            query_embedings = self.encoder(
                input_ids = queries_tokens["input_ids"].squeeze(1),
                attention_mask=queries_tokens["attention_mask"].squeeze(1) ,
                return_dict=False
            )[0][:, 0, :].unsqueeze(1).contiguous()
            assert query_embedings.shape == (batch_size, 1, self.hidden_size)

        if self.use_cuda:
            candidates_tokens["input_ids"] = candidates_tokens["input_ids"].to(self.device, non_blocking=True)
            candidates_tokens["attention_mask"] = candidates_tokens["attention_mask"].to(self.device, non_blocking=True)
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_cuda):
            candidate_embeddings = self.encoder(
                input_ids=candidates_tokens["input_ids"].reshape(-1, max_length),
                attention_mask=candidates_tokens["attention_mask"].reshape(-1, max_length),
                return_dict=False
                )[0][:, 0, :].reshape(batch_size, topk, -1).transpose(1, 2).contiguous()

            assert candidate_embeddings.shape == (batch_size, self.hidden_size, self.topk), f'Current candidate embs shape: {candidate_embeddings.shape}'

        scores = torch.bmm(query_embedings, candidate_embeddings).squeeze(1)
        return scores

    def get_loss(self, scores, targets):
        if self.use_cuda:
            targets =targets.cuda()
        loss = self.marginal_nll(scores,targets)
        return loss

    def save_state(self, path):
        self.encoder.save_pretrained(path)

    def marginal_nll(self, score, target):
        score = score.float()
        target = target.float()
        predict = torch.softmax(score, dim=-1)
        loss = predict * target
        loss = loss.sum(dim=-1)                   # sum all positive scores
        loss = loss[loss > 0]                     # filter sets with at least one positives
        loss = torch.clamp(loss, min=1e-9, max=1) # for numerical stability
        loss = -torch.log(loss)                   # for negative log likelihood
        if len(loss) == 0:
            loss = loss.sum()                     # will return zero loss
        else:
            loss = loss.mean()
        return loss


class FaissIndex():
    def __init__(self,  encoder, config, tokens):
        self.encoder = encoder


        self.use_cuda = config.use_cuda
        self.use_fp16 = config.use_fp16
        self.hidden_size = getattr(getattr(encoder, "config", None), "hidden_size", None)
        assert self.hidden_size is not None



        self.tokens  = tokens

        self.max_length = config.max_length
        self.topk = config.topk
        self.amp_dtype = config.amp_dtype



        self.faiss_build_batch_size = config.faiss_build_batch_size
        self.faiss_search_batch_size = config.faiss_search_batch_size


        if self.use_cuda:
            gpu_resources = faiss.StandardGpuResources()
            index_conf = faiss.GpuIndexFlatConfig()
            index_conf.device = torch.cuda.current_device()
            index_conf.useFloat16 = bool(self.use_fp16)
            index = faiss.GpuIndexFlatIP(gpu_resources, self.hidden_size, index_conf)
            self.device = "cuda"
        else:
            index = faiss.IndexFlatIP(self.hidden_size)
            self.device = "cpu"

        assert index is not None
        self.index = index



    def build_index(self):
        (tokens_size, max_length ) = self.tokens["dictionary_inputs"].shape

        assert tokens_size > 0 and max_length == self.max_length
        batch_size=self.faiss_build_batch_size


        self.encoder.eval()
        with torch.inference_mode():
            for start in tqdm(range(0,tokens_size, batch_size), desc="embeding and building faiss", unit="bach"):
                end = min(start+batch_size, tokens_size)
                chunk_input_ids = self.tokens["dictionary_inputs"][start:end]
                chunk_att_mask = self.tokens["dictionary_attention"][start:end]
                chunk_input_ids = torch.from_numpy(chunk_input_ids).to(device=self.device, dtype=torch.long)
                chunk_att_mask = torch.from_numpy(chunk_att_mask).to(device=self.device, dtype=torch.long)
                if self.use_fp16:
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                        out_chunk = self.encoder(
                            input_ids= chunk_input_ids,
                            attention_mask=chunk_att_mask,
                            return_dict=True
                        )[0][:,0]
                else:
                    out_chunk = self.encoder(
                        input_ids= chunk_input_ids,
                        attention_mask=chunk_att_mask,
                        return_dict=True
                    )[0][:,0] # cls (chunk_size, hidden_size)

                assert out_chunk is not None
                out_chunk = out_chunk.contiguous()
                self.index.add(out_chunk)
                del out_chunk, chunk_input_ids,chunk_att_mask

    def search_candidates(self):
        (tokens_size, max_length ) = self.tokens["query_inputs"].shape
        assert tokens_size > 0 and max_length == self.max_length

        batch_size = self.faiss_search_batch_size
        candidates_idxs = []
        self.encoder.eval()
        with torch.inference_mode():
            for start in tqdm(range(0,tokens_size, batch_size), desc="embeding and search index for candidates", unit="bach"):
                end = min(start+batch_size, tokens_size)
                chunk_input_ids = self.tokens["query_inputs"][start:end]
                chunk_att_mask = self.tokens["query_attention"][start:end]

                chunk_input_ids = torch.from_numpy(chunk_input_ids).to(device=self.device, dtype=torch.long)
                chunk_att_mask = torch.from_numpy(chunk_att_mask).to(device=self.device, dtype=torch.long)

                if self.use_fp16:
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                        out_chunk = self.encoder(
                            input_ids= chunk_input_ids,
                            attention_mask=chunk_att_mask,
                            return_dict=True
                        )[0][:,0] # cls (chunk_size, hidden_size)
                else:
                    out_chunk = self.encoder(
                        input_ids= chunk_input_ids,
                        attention_mask=chunk_att_mask,
                        return_dict=True
                    )[0][:,0] # cls (chunk_size, hidden_size)

                assert out_chunk is not None
                out_chunk = out_chunk.contiguous()


                _, chunk_cand_idxs = self.index.search(out_chunk, self.topk)


                if self.use_cuda:
                    chunk_cand_idxs = chunk_cand_idxs.cpu().numpy()
                candidates_idxs.append(chunk_cand_idxs)
                del chunk_cand_idxs, out_chunk
        return np.vstack(candidates_idxs)



def load_mmap_shape(base):
    with open(base+".json") as f:
        meta = json.load(f)
    return tuple(meta["shape"])



class MyDataSet(torch.utils.data.Dataset):
        def __init__(self, confs):
            self.max_length = confs.max_length
            self.topk = confs.topk
            self.d_candidate_idxs = None
            self.labels_per_query = None


            queries_input_ids_mmap_path = confs.queries_mmap_base + confs.tokens_inputs_file_suffix
            queries_attention_mask_mmap_path = confs.queries_mmap_base + confs.tokens_attentions_file_suffix
            queries_cuis_path = confs.queries_mmap_base + confs.cuis_file_suffix

            dictionary_input_ids_mmap_path = confs.dictionary_mmap_base + confs.tokens_inputs_file_suffix
            dictionary_attention_mask_mmap_path = confs.dictionary_mmap_base + confs.tokens_attentions_file_suffix
            dictionary_cuis_path = confs.dictionary_mmap_base + confs.cuis_file_suffix

            self.tokens = {
                "dictionary_inputs":  np.memmap(
                    dictionary_input_ids_mmap_path,
                    mode="r+",
                    dtype=np.int32,
                    shape=load_mmap_shape(confs.dictionary_mmap_base) 
                ),
                "dictionary_attention":  np.memmap(
                    dictionary_attention_mask_mmap_path,
                    mode="r+",
                    dtype=np.int32,
                    shape=load_mmap_shape(confs.dictionary_mmap_base) 
                ),
                "query_inputs":  np.memmap(
                    queries_input_ids_mmap_path,
                    mode="r+",
                    dtype=np.int32,
                    shape=load_mmap_shape(confs.queries_mmap_base) 
                ),
                "query_attention":  np.memmap(
                    queries_attention_mask_mmap_path,
                    mode="r+",
                    dtype=np.int32,
                    shape=load_mmap_shape(confs.queries_mmap_base) 
                )
            }
            self.queries_cuis = np.load(queries_cuis_path)
            self.dictionary_cuis = np.load(dictionary_cuis_path)

            self.cui_to_idx_dictionary_map = {}
            for dictionary_idx, dictionary_cui in enumerate(self.dictionary_cuis):
                self.cui_to_idx_dictionary_map.setdefault(dictionary_cui, []).append(dictionary_idx)


            save_pkl(self.cui_to_idx_dictionary_map, "./data/draft/cui_to_idx_dictionary_map.pkl")
            assert self.tokens["query_inputs"].shape == self.tokens["query_attention"].shape == (len(self.queries_cuis),self.max_length)
            assert self.tokens["dictionary_inputs"].shape == self.tokens["dictionary_attention"].shape == (len(self.dictionary_cuis),self.max_length)

        def __getitem__(self,query_idx):
            assert (self.candidate_idxs is not None)
            assert self.labels_per_query is not None


            query_tokens = {
                "input_ids": torch.from_numpy(self.tokens["query_inputs"][query_idx]),
                "attention_mask": torch.from_numpy(self.tokens["query_attention"][query_idx]),
            }

            query_candidates_idxs = self.candidate_idxs[query_idx]
            query_candidates_idxs = np.array(query_candidates_idxs)

            query_candidates_tokens = {
                "input_ids": torch.from_numpy(self.tokens["dictionary_inputs"][query_candidates_idxs]),
                "attention_mask": torch.from_numpy(self.tokens["dictionary_attention"][query_candidates_idxs]),
            }



            labels = self.labels_per_query[query_idx]

            assert len(query_candidates_idxs) == self.topk == len(set(query_candidates_idxs)) == len(labels)
            assert query_candidates_tokens["input_ids"].shape == (self.topk, self.max_length), f'query_tokens shape is wrong {query_tokens["input_ids"].shape}'

            return (query_tokens, query_candidates_tokens), labels


        def __len__(self):
            return len(self.queries_cuis)


        def set_dense_candidate_idxs(self, candidate_indexs):
            """
                candidate_indexes that we got from faiss index
                we are updating the candidates array so every query has at least 1 golden candidate
                we are setting self.labels_per_query having shape (queries_num, topk) having labels (1.0, 0.0)
            """
            candidate_idxs_old = candidate_indexs


            #set labels for after in getitem, and set new_candidates which will have golden candidate
            labels_per_query = []
            new_candidates = []
            for query_idx, cand_idxs in enumerate(candidate_idxs_old):
                query_cui = self.queries_cuis[query_idx]

                assert query_cui in set(self.cui_to_idx_dictionary_map.keys()), f"query_idx: {query_idx}, query_cui: {query_cui}, len(map_cui_to_idxdictionary): {len(self.cui_to_idx_dictionary_map)} "
                golden_query_dictionary_idx = self.cui_to_idx_dictionary_map[query_cui]


                labels = np.fromiter(
                    (
                        1.0 if query_cui == self.dictionary_cuis[candidate_idx]
                        else 0.0
                        for candidate_idx in cand_idxs
                    ),
                    dtype=np.float32,
                    count=len(cand_idxs)
                )
                if labels.max() == 0.0 and golden_query_dictionary_idx:
                    replace_pos = 0
                    cand_idxs[replace_pos] = golden_query_dictionary_idx[0]
                    labels[replace_pos] = 1.0

                labels_per_query.append(labels)
                new_candidates.append(cand_idxs)

            self.candidate_idxs = np.array(new_candidates)
            self.labels_per_query = np.array(labels_per_query)



class MetricsLogger:
    def __init__(self, logger, confs, tag="train"):
        self.use_cuda = confs.use_cuda
        self.device = "cuda" if self.use_cuda else "cpu"
        self.logger = logger
        self.tag = tag
        self.process = psutil.Process(os.getpid())
        
        self.cpu_memory_used = 0.0
        self.messages = []
        self.one_time_events_set = set()

    def current_cpu_mem_usage(self):
        rss = self.process.memory_info().rss / (1024 * 2)
        self.cpu_memory_used = rss
        return rss


    def current_gpu_mem_usage(self):
        if self.use_cuda:
            free = torch.cuda.mem_get_info()[0] / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            return (free, total)
        return (0.0,0.0)


    def current_gpu_stats(self):
        """
            alloc (current allocated memory in MB): Memory currently allocated by tensors.
            alloc_peak (peak allocated memory in MB): Highest memory allocated by tensors since the program start or last reset.
            res (current reserved memory in MB): Memory reserved by the caching allocator (includes allocated plus cached blocks).
            res_peak (peak reserved memory in MB): Highest reserved memory since the program start or last reset.

        """
        if not self.use_cuda:
            return (None, None, None, None)
        torch.cuda.synchronize(self.device)
        alloc = torch.cuda.memory_allocated(self.device) / (1024**2)
        alloc_peak = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        res = torch.cuda.memory_reserved(self.device) / (1024**2)
        res_peak = torch.cuda.max_memory_reserved(self.device) / (1024**2)
        return (alloc, alloc_peak, res, res_peak)



    def log_event(self, event_tag, t0=None, log_immediate=True, first_iteration_only=False, only_elapsed_time=False):
        if first_iteration_only and event_tag in self.one_time_events_set:
            return True


        self.one_time_events_set.add(event_tag)
        msg = f"[{self.tag}-{event_tag}] "


        if t0:
            elapsed = time.time() - t0
            msg += f" | elapsed time: {elapsed:.5f}seconds "


        if only_elapsed_time:
            return self.logger.info(f"\n{msg}") if log_immediate else self.messages.append(f"\n{msg}")

        msg += f" | CPU Memory usage: {self.current_cpu_mem_usage():.1f}MB "
        if self.use_cuda:
            (free, total) = self.current_gpu_mem_usage()
            msg += f" | GPU memory total/free: {total:.1f}/{free:.1f}MB"
            (alloc, alloc_peak, res, res_peak) = self.current_gpu_stats()
            msg += f" | CUDA: allocated/peak: {alloc:.1f}/{alloc_peak:.1f}MB, reserved/peak {res:.1f}/{res_peak:.1f}MB"

        return self.logger.info(f"\n{msg}") if log_immediate else self.messages.append(f"\n{msg}")

