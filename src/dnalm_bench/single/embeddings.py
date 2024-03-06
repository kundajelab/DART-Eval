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
from ..utils import onehot_to_chars



class SimpleEmbeddingExtractor:
    _idx_mode = "variable"

    @staticmethod
    def _offsets_to_indices(offsets, seqs):
        gather_idx = np.zeros((seqs.shape[0], seqs.shape[1]), dtype=np.uint32)
        for i, offset in enumerate(offsets):
            for j, (start, end) in enumerate(offset):
                gather_idx[i,start:end] = j
        
        return gather_idx

    def extract_embeddings(self, dataset, out_path, progress_bar=False):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        with h5py.File(out_path + ".tmp", "w") as out_f:
            seq_grp = out_f.create_group("seq")

            start = 0
            for seqs in tqdm(dataloader, disable=(not progress_bar)):
                end = start + len(seqs)

                seq_tokens, seq_offsets = self.tokenize(seqs)

                seq_token_emb = self.model_fwd(seq_tokens)

                if self._idx_mode == "variable":
                    seq_indices = self._offsets_to_indices(seq_offsets, seqs)
                    seq_indices_dset = seq_grp.require_dataset("idx_var", (len(dataset), seq_indices.shape[1]), dtype=np.uint32)
                    seq_indices_dset[start:end] = seq_indices

                elif (start == 0) and (self._idx_mode == "fixed"):
                    seq_indices = self._offsets_to_indices(seq_offsets, seqs)
                    seq_indices_dset = seq_grp.create_dataset("idx_fix", data=seq_indices, dtype=np.uint32)

                seq_grp.create_dataset(f"emb_{start}_{end}", data=seq_token_emb.numpy(force=True))

                start = end

        os.rename(out_path + ".tmp", out_path)


# class HFSimpleEmbeddingExtractor(HFEmbeddingExtractor):
#     def __init__(self, tokenizer, model, batch_size, num_workers, device):
#         super().__init__(tokenizer, model, batch_size, num_workers, device)

#     def extract_embeddings(self, dataset, out_path, progress_bar=False):
#         dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

#         with h5py.File(out_path, "w") as out_f:
#             start = 0
#             for seqs in tqdm(dataloader, disable=(not progress_bar)):
#                 end = start + len(seqs)

#                 seq_tokens, seq_offsets = self.tokenize(seqs)

#                 seq_token_emb = self.model_fwd(seq_tokens)

#                 seq_embeddings = self.detokenize(seqs, seq_token_emb, seq_offsets)
#                 # print(seq_embeddings.shape) ####

#                 seq_embeddings_dset = out_f.require_dataset("seq_emb", (len(dataset), seq_embeddings.shape[1], seq_embeddings.shape[2]), 
#                                                             chunks=seq_embeddings.shape, dtype=np.float32, compression="gzip", compression_opts=1)
#                 seq_embeddings_dset[start:end] = seq_embeddings.numpy(force=True)

#                 start = end

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

    
class DNABERT2EmbeddingExtractor(HFEmbeddingExtractor, SimpleEmbeddingExtractor):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"zhihan1996/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        config = BertConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_config(config)
        super().__init__(tokenizer, model, batch_size, num_workers, device)


class MistralDNAEmbeddingExtractor(HFEmbeddingExtractor, SimpleEmbeddingExtractor):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"RaphaelMourad/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

class GENALMEmbeddingExtractor(HFEmbeddingExtractor, SimpleEmbeddingExtractor):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"AIRI-Institute/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

class NucleotideTransformerEmbeddingExtractor(HFEmbeddingExtractor, SimpleEmbeddingExtractor):
    _idx_mode = "fixed"

    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"InstaDeepAI/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model =  AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

    def tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer(seqs_str, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]
        # print(tokens.shape) ####

        return tokens, None

    @staticmethod
    def _offsets_to_indices(offsets, seqs):
        seq_len = seqs.shape[1]
        inds = np.zeros(seq_len, dtype=np.int32)
        # seq_len_contig = (seq_len // 6) * 6
        for i in range(seq_len // 6):
            inds[i*6:(i+1)*6] = i + 1
        inds[(i+1)*6:] = np.arange(i+2, i+(seq_len%6)+2)

        return inds

class HyenaDNAEmbeddingExtractor(HFEmbeddingExtractor, SimpleEmbeddingExtractor):
    _idx_mode = "fixed"

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

    @staticmethod
    def _offsets_to_indices(offsets, seqs):
        slice_idx = [0, seqs.shape[1]]
        
        return np.array(slice_idx)


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

class GenaLMVariantEmbeddingExtractor(HFVariantEmbeddingExtractor):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"AIRI-Institute/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)


class HyenaDNAVariantEmbeddingExtractor(HFVariantEmbeddingExtractor):
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

