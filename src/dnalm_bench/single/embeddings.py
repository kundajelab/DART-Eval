import os
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForCausalLM, BertConfig
from scipy.stats import wilcoxon
from tqdm import tqdm
import h5py
from ..embeddings import HFEmbeddingExtractor


class HFSimpleEmbeddingExtractor(HFEmbeddingExtractor):
    def __init__(self, tokenizer, model, batch_size, num_workers, device):
        super().__init__(tokenizer, model, batch_size, num_workers, device)

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

class HFVariantEmbeddingExtractor(HFEmbeddingExtractor):
    def __init__(self, tokenizer, model, batch_size, num_workers, device):
        super().__init__(tokenizer, model, batch_size, num_workers, device)

    def extract_embeddings(self, dataset, out_path, progress_bar=False):
        # breakpoint()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        with h5py.File(out_path, "w") as out_f:
            start = 0
            print(dataloader)
            for ref, alt in tqdm(dataloader, disable=(not progress_bar)):
                if torch.all(ref == 0) and torch.all(alt==0):
                    continue

                end = start + len(ref)

                ref_tokens, ref_offsets = self.tokenize(ref)
                alt_tokens, alt_offsets = self.tokenize(alt)

                ref_token_emb = self.model_fwd(ref_tokens)
                alt_token_emb = self.model_fwd(alt_tokens)

                ref_embeddings = self.detokenize(ref, ref_token_emb, ref_offsets)
                alt_embeddings = self.detokenize(alt, alt_token_emb, alt_offsets)

                ref_embeddings_dset = out_f.require_dataset("ref_emb", (len(dataset), ref_embeddings.shape[1], ref_embeddings.shape[2]),
                                                            chunks=ref_embeddings.shape, dtype=np.float32, compression="gzip", compression_opts=1)
                alt_embeddings_dset = out_f.require_dataset("alt_emb", (len(dataset), alt_embeddings.shape[1], alt_embeddings.shape[2]),
                        chunks = alt_embeddings.shape, dtype=np.float32, compression="gzip", compression_opts=1)

                ref_embeddings_dset[start:end] = ref_embeddings.numpy(force=True)

                alt_embeddings_dset[start:end] = alt_embeddings.numpy(force=True)

                start = end

    
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
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

class DNABERT2VariantEmbeddingExtractor(HFVariantEmbeddingExtractor):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"zhihan1996/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        config = BertConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_config(config)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

class MistralDNAVariantEmbeddingExtractor(HFVariantEmbeddingExtractor):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"RaphaelMourad/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

