import os
from abc import ABCMeta, abstractmethod
import hashlib
import shutil
import importlib
import json
import warnings
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig, AutoModel
from tqdm import tqdm
import polars as pl
import h5py
import pyfaidx
import pyBigWig
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

from ..finetune import HFClassifierModel, LoRAModule
from ..utils import onehot_to_chars, one_hot_encode, NoModule, log1mexp



def profile_model_resources(dataset, model, batch_size, num_batches_warmup, out_path, num_workers, prefetch_factor, device, progress_bar=False, seed=0, num_batches_record=np.inf):
    num_batches_total = num_batches_warmup + num_batches_record
    
    num_params = sum(p.numel() for p in model.parameters())
    
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, prefetch_factor=prefetch_factor, drop_last=True)

    model.train()
    bwd_mem = []
    bwd_time = []
    for i, (seq, track) in enumerate(tqdm(dataloader, disable=(not progress_bar), desc="bwd", ncols=120)):
        if i >= num_batches_total:
            break

        torch.cuda.reset_peak_memory_stats(device=device)
        track = track.to(device)
        model.zero_grad()
        start = time.time()
        loss = model(seq).sum()
        loss.backward()
        torch.cuda.synchronize(device=device)
        end = time.time()

        mem_usage = torch.cuda.max_memory_allocated(device=device)
        time_elapsed = end - start

        if i >= num_batches_warmup:
            bwd_mem.append(mem_usage)
            bwd_time.append(time_elapsed)

    bwd_mem_mean = np.mean(bwd_mem)
    bwd_mem_std = np.std(bwd_mem)
    bwd_time_mean = np.mean(bwd_time)
    bwd_time_std = np.std(bwd_time)
    
    model.eval()
    with torch.no_grad():
        fwd_mem = []
        fwd_time = []
        
        for i, (seq, track) in enumerate(tqdm(dataloader, disable=(not progress_bar), desc="fwd", ncols=120)):
            if i >= num_batches_total:
                break

            torch.cuda.reset_peak_memory_stats(device=device)
            track = track.to(device)
            start = time.time()
            _ = model(seq)
            torch.cuda.synchronize(device=device)
            end = time.time()

            mem_usage = torch.cuda.max_memory_allocated(device=device)
            time_elapsed = end - start

            if i >= num_batches_warmup:
                fwd_mem.append(mem_usage)
                fwd_time.append(time_elapsed)

        fwd_mem_mean = np.mean(fwd_mem)
        fwd_mem_std = np.std(fwd_mem)
        fwd_time_mean = np.mean(fwd_time)
        fwd_time_std = np.std(fwd_time)

    metrics = {
        "num_params": num_params,
        "fwd_mem_mean": fwd_mem_mean,
        "fwd_mem_std": fwd_mem_std,
        "fwd_time_mean": fwd_time_mean,
        "fwd_time_std": fwd_time_std,
        "bwd_mem_mean": bwd_mem_mean,
        "bwd_mem_std": bwd_mem_std,
        "bwd_time_mean": bwd_time_mean,
        "bwd_time_std": bwd_time_std
    }
        
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


class DNABERT2Model(HFClassifierModel):
    def __init__(self, model_name, num_labels):
        model_name = f"zhihan1996/{model_name}"
        with NoModule("triton"):
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            config = BertConfig.from_pretrained(model_name)
            config.num_labels = num_labels
            model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, config=config)

        super().__init__(tokenizer, model)


class MistralDNAModel(HFClassifierModel):
    def __init__(self, model_name, num_labels):
        model_name = f"RaphaelMourad/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, num_labels=num_labels)
        model.config.pad_token_id = tokenizer.pad_token_id
        
        super().__init__(tokenizer, model)


class GENALMModel(HFClassifierModel):
    def __init__(self, model_name, num_labels):
        model_name = f"AIRI-Institute/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_base = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        gena_module_name = model_base.__class__.__module__
        cls = getattr(importlib.import_module(gena_module_name), 'BertForSequenceClassification')
        model = cls.from_pretrained(model_name, num_labels=num_labels)

        super().__init__(tokenizer, model)


class NucleotideTransformerModel(HFClassifierModel):
    def __init__(self, model_name, num_labels):
        model_name = f"InstaDeepAI/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, num_labels=num_labels)

        super().__init__(tokenizer, model)

    def _tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer(seqs_str, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]

        return tokens.to(self.device), None


class HyenaDNAModel(HFClassifierModel):
    def __init__(self, model_name, num_labels):
        model_name = f"LongSafari/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, num_labels=num_labels)

        super().__init__(tokenizer, model)

    def _tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer(seqs_str, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]

        return tokens.to(self.device), None
    
    def forward(self, seqs):
        tokens, _ = self._tokenize(seqs)

        torch_outs = self.model(tokens)
        logits = torch_outs.logits

        return logits


class CaduceusModel(HFClassifierModel):
    def __init__(self, model_name, num_labels):
        model_name = f"kuleshov-group/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, num_labels=num_labels)

        super().__init__(tokenizer, model)

    def _tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer(seqs_str, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]

        return tokens.to(self.device), None
    
    def forward(self, seqs):
        tokens, _ = self._tokenize(seqs)

        torch_outs = self.model(tokens)
        logits = torch_outs.logits

        return logits