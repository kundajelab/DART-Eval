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

from ..components import PairedControlDataset
from ...utils import onehot_to_chars
from ...embeddings import HFEmbeddingExtractor

class HFPairedControlEmbeddingExtractor(HFEmbeddingExtractor):
    def __init__(self, tokenizer, model, batch_size, num_workers, device):
        super().__init__(tokenizer, model, batch_size, num_workers, device)

    def extract_embeddings(self, dataset, out_path, progress_bar=False):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        with h5py.File(out_path, "w") as out_f:
            start = 0
            for seqs, ctrls in tqdm(dataloader, disable=(not progress_bar)):
                end = start + len(seqs)

                seq_tokens, seq_offsets = self.tokenize(seqs)
                ctrl_tokens, ctrl_offsets = self.tokenize(ctrls)

                seq_token_emb = self.model_fwd(seq_tokens)
                ctrl_token_emb = self.model_fwd(ctrl_tokens)

                seq_embeddings = self.detokenize(seqs, seq_token_emb, seq_offsets)
                ctrl_embeddings = self.detokenize(ctrls, ctrl_token_emb, ctrl_offsets)
                # print(seq_embeddings.shape) ####

                seq_embeddings_dset = out_f.require_dataset("seq_emb", (len(dataset), seq_embeddings.shape[1], seq_embeddings.shape[2]), 
                                                            chunks=seq_embeddings.shape, dtype=np.float32, compression="gzip", compression_opts=1)
                seq_embeddings_dset[start:end] = seq_embeddings.numpy(force=True)

                ctrl_embeddings_dset = out_f.require_dataset("ctrl_emb", (len(dataset), ctrl_embeddings.shape[1], ctrl_embeddings.shape[2]), 
                                                             chunks=ctrl_embeddings.shape, dtype=np.float32, compression="gzip", compression_opts=1)
                ctrl_embeddings_dset[start:end] = ctrl_embeddings.numpy(force=True)

                start = end

class DNABERT2EmbeddingExtractor(HFPairedControlEmbeddingExtractor):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"zhihan1996/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        config = BertConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_config(config)
        super().__init__(tokenizer, model, batch_size, num_workers, device)


class GenaLMEmbeddingExtractor(HFPairedControlEmbeddingExtractor):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"AIRI-Institute/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)


class HyenaDNAEmbeddingExtractor(HFPairedControlEmbeddingExtractor):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"LongSafari/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
        model =  AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

    def tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer(seqs_str, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]

        return tokens, None

    def detokenize(self, seqs, token_embeddings, _):
        seq_embeddings = token_embeddings[:,:seqs.shape[1],:]

        return seq_embeddings