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


class SimpleSequenceEmbeddingExtractor(metaclass=ABCMeta):
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

    def extract_embeddings(self, dataset, out_path, progress_bar=False):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        with h5py.File(out_path, "w") as out_f:
            start = 0
            for seqs in tqdm(dataloader, disable=(not progress_bar)):
                end = start + len(seqs)

                seq_tokens, seq_offsets = self.tokenize(seqs)

                seq_token_emb = self.model_fwd(seq_tokens)

                seq_embeddings = self.detokenize(seqs, seq_token_emb, seq_offsets)
                # print(seq_embeddings.shape) ####

                seq_embeddings_dset = out_f.require_dataset("seq_emb", (len(dataset), seq_embeddings.shape[1], seq_embeddings.shape[2]), 
                                                            chunks=seq_embeddings.shape, dtype=np.float32, compression="gzip", compression_opts=1)
                seq_embeddings_dset[start:end] = seq_embeddings.numpy(force=True)

                start = end
    

class HFSimpleEmbeddingExtractor(SimpleSequenceEmbeddingExtractor):
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
        # print(token_embeddings.shape) ####
        # print(len(offsets[0])) ####
        gather_idx = torch.zeros((seqs.shape[0], seqs.shape[1], 1), dtype=torch.long)
        for i, offset in enumerate(offsets):
            for j, (start, end) in enumerate(offset):
                gather_idx[i,start:end,:] = j

        gather_idx = gather_idx.expand(-1,-1,token_embeddings.shape[2]).to(self.device)
        seq_embeddings = torch.gather(token_embeddings, 1, gather_idx)
        # print(seq_embeddings.shape) ####

        return seq_embeddings


class DNABERT2EmbeddingExtractor(HFSimpleEmbeddingExtractor):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"zhihan1996/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        config = BertConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_config(config)
        super().__init__(tokenizer, model, batch_size, num_workers, device)


class MistralDNAEmbeddingExtractor(HFSimpleEmbeddingExtractor):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"RaphaelMourad/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

