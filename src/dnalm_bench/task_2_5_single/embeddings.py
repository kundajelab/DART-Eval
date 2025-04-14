import os
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForCausalLM, BertConfig, AutoConfig
from scipy.stats import wilcoxon
from tqdm import tqdm
import h5py
from ..embeddings import HFEmbeddingExtractor, SequenceBaselineEmbeddingExtractor
from ..utils import onehot_to_chars, NoModule



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


class HFVariantEmbeddingExtractor(HFEmbeddingExtractor):
    _idx_mode = "variable"

    def __init__(self, tokenizer, model, batch_size, num_workers, device):
        super().__init__(tokenizer, model, batch_size, num_workers, device)

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
            allele1_grp = out_f.create_group("allele1")
            allele2_grp = out_f.create_group("allele2")

            start = 0
            for allele1, allele2 in tqdm(dataloader, disable=(not progress_bar)): # shape = batch_size x 500 x 4
                if torch.all(allele1 == 0) and torch.all(allele2==0):
                    continue
                end = start + len(allele1)

                allele1_tokens, allele1_offsets = self.tokenize(allele1)
                allele2_tokens, allele2_offsets = self.tokenize(allele2)

                allele1_token_emb = self.model_fwd(allele1_tokens)
                allele2_token_emb = self.model_fwd(allele2_tokens)
                if self._idx_mode == "variable":
                    allele1_indices = self._offsets_to_indices(allele1_offsets, allele1)
                    allele1_indices_dset = allele1_grp.require_dataset("idx_var", (len(dataset), allele1_indices.shape[1]), dtype=np.uint32)
                    allele1_indices_dset[start:end] = allele1_indices

                    allele2_indices = self._offsets_to_indices(allele2_offsets, allele2)
                    allele2_indices_dset = allele2_grp.require_dataset("idx_var", (len(dataset), allele2_indices.shape[1]), dtype=np.uint32)
                    allele2_indices_dset[start:end] = allele2_indices

                elif (start == 0) and (self._idx_mode == "fixed"):
                    allele1_indices = self._offsets_to_indices(allele1_offsets, allele1)
                    allele1_indices_dset = allele1_grp.create_dataset("idx_fix", data=allele1_indices, dtype=np.uint32)
                    allele2_indices = self._offsets_to_indices(allele2_offsets, allele2)
                    allele2_indices_dset = allele2_grp.create_dataset("idx_fix", data=allele2_indices, dtype=np.uint32)

                allele1_grp.create_dataset(f"emb_{start}_{end}", data=allele1_token_emb.numpy(force=True))
                allele2_grp.create_dataset(f"emb_{start}_{end}", data=allele2_token_emb.numpy(force=True))

                start = end
        os.rename(out_path + ".tmp", out_path)      


class SequenceBaselineSimpleEmbeddingExtractor(SequenceBaselineEmbeddingExtractor, SimpleEmbeddingExtractor):
    _idx_mode = "fixed"

    @staticmethod
    def _offsets_to_indices(offsets, seqs):
        slice_idx = [0, seqs.shape[1]]
        
        return np.array(slice_idx) 

    
class DNABERT2EmbeddingExtractor(HFEmbeddingExtractor, SimpleEmbeddingExtractor):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"zhihan1996/{model_name}"
        with NoModule("triton"):
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            config = BertConfig.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForMaskedLM.from_pretrained(model_name, config=config, trust_remote_code=True)

        super().__init__(tokenizer, model, batch_size, num_workers, device)

    def model_fwd(self, tokens):
        tokens = tokens.to(device=self.device)
        with torch.no_grad():
            torch_outs = self.model(
                tokens,
                output_hidden_states=True
            )
            embs = torch_outs.hidden_states

        return embs


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
        print(model)
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

class CaduceusEmbeddingExtractor(HFEmbeddingExtractor, SimpleEmbeddingExtractor):
    _idx_mode = "fixed"   
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"kuleshov-group/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
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
    
class HyenaDNAUntrainedEmbeddingExtractor(HFEmbeddingExtractor, SimpleEmbeddingExtractor):
    _idx_mode = "fixed"

    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"LongSafari/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model =  AutoModelForCausalLM.from_config(config, trust_remote_code=True)
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
        with NoModule("triton"):
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            config = BertConfig.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForMaskedLM.from_pretrained(model_name, config=config, trust_remote_code=True)

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

    def detokenize(self, seqs, token_embeddings, _):
        seq_embeddings = token_embeddings[:,:seqs.shape[1],:]

        return seq_embeddings
    
    @staticmethod
    def _offsets_to_indices(offsets, seqs):
        slice_idx = [0, seqs.shape[1]]
        
        return np.array(slice_idx)
    
class NucleotideTransformerVariantEmbeddingExtractor(HFVariantEmbeddingExtractor):
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


def encode_sequence(sequence): 
    encoded_sequence = [MAPPING.get(nucleotide, 4) for nucleotide in sequence]
    return encoded_sequence

def encode_sequence_batch(seq_batch):
   return [encode_sequence(seq) for seq in seq_batch]

class RegulatoryLMEmbeddingExtractor(SimpleEmbeddingExtractor, HFEmbeddingExtractor):
    _idx_mode = "fixed"
    def __init__(self, model, batch_size, num_workers, device, category=12, mask_token=5, seq_input_size=350, model_input_size=350):
        tokenizer = None
        self.category = category
        self.mask_token_override = mask_token
        self.seq_input_size = seq_input_size
        self.model_input_size = model_input_size
        self.mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        super().__init__(tokenizer, model, batch_size, num_workers, device)

    def encode_sequence(self, sequence): 
        encoded_sequence = [self.mapping.get(nucleotide, 4) for nucleotide in sequence]
        return encoded_sequence

    def encode_sequence_batch(self, seq_batch):
        return [self.encode_sequence(seq) for seq in seq_batch]

    @staticmethod
    def _offsets_to_indices(offsets, seqs):
        slice_idx = [0, seqs.shape[1] - 1]
        
        return np.array(slice_idx)

    @property
    def start_token(self):
        return None
    
    @property
    def end_token(self):
        return None

    def tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        tokens = torch.tensor(self.encode_sequence_batch(seqs_str))
        return tokens, None


    def model_fwd(self, tokens):
        tokens = tokens.to(self.device)
        if self.category is not None:
            category_tensor = torch.tensor([self.category]).to(device=self.device)
        else:
            category_tensor = self.category
        with torch.no_grad():
            if self.seq_input_size == self.model_input_size:
                embs = self.model.embed(tokens, category_tensor)
            #Else case - adapting to different input sizes
            #We basically break up the sequence into chunks of the model input length
            #Any remaining tokens are added by predicting the very end of the sequence and only concatenating the embeddings for previously unpredicted tokens
            else:
                full_partitions, remainder = self.seq_input_size // self.model_input_size, self.seq_input_size % self.model_input_size
                for part in range(full_partitions):
                    curr_tokens = tokens[:,part * self.model_input_size : part * self.model_input_size + self.model_input_size]
                    if part == 0:
                        embs = self.model.embed(curr_tokens, category_tensor)
                    else:
                        curr_embs = self.model.embed(curr_tokens, category_tensor)
                        embs = torch.cat((embs, curr_embs), dim=1)
                #To account for the stragglers, we predict the very end of the sequence but only concatenate the stragglers
                final_pred = self.model.embed(tokens[:,-1*self.model_input_size:], category_tensor)
                embs = torch.cat((embs, final_pred[:,-1*remainder:]), dim=1)
        return embs

