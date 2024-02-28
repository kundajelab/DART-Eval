import os
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForCausalLM, BertConfig
from scipy.stats import wilcoxon
from tqdm import tqdm
import h5py

from .utils import onehot_to_chars


class EmbeddingExtractor(metaclass=ABCMeta):
    def __new__(cls, *args, **kwargs):

        return super().__new__(cls)

    @abstractmethod
    def __init__(self, batch_size, num_workers, device):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    @abstractmethod
    def tokenize(self, seqs):
        pass

    @abstractmethod
    def model_fwd(self, tokens, attention_mask):
        pass

    @abstractmethod
    def detokenize(self, seqs, token_embeddings, offsets):
        pass

class HFEmbeddingExtractor(EmbeddingExtractor):
    def __init__(self, tokenizer, model, batch_size, num_workers, device):
        self.tokenizer = tokenizer
        self.model = model
        self.model.to(device)
        model.eval()
        super().__init__(batch_size, num_workers, device)

    def tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer(seqs_str, return_tensors="pt", padding=True, return_offsets_mapping=True)
        tokens = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        return tokens, offsets

    def model_fwd(self, tokens):
        tokens = tokens.to(device=self.device)
        with torch.no_grad():
            torch_outs = self.model(
                tokens,
                output_hidden_states=True
            )
            embs = torch_outs.hidden_states[-1]

        return embs

    def detokenize(self, seqs, token_embeddings, offsets):
        gather_idx = torch.zeros((seqs.shape[0], seqs.shape[1], 1), dtype=torch.long)
        for i, offset in enumerate(offsets):
            for j, (start, end) in enumerate(offset):
                gather_idx[i,start:end,:] = j

        gather_idx = gather_idx.expand(-1,-1,token_embeddings.shape[2]).to(self.device)
        seq_embeddings = torch.gather(token_embeddings, 1, gather_idx)

        return seq_embeddings
