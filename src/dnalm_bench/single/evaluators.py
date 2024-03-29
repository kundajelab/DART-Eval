from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForCausalLM, BertConfig
from scipy.stats import wilcoxon
from tqdm import tqdm
from ..utils import onehot_to_chars
import polars as pl

class LikelihoodEvaluator(metaclass=ABCMeta):
    def __init__(self, tokenizer, model, batch_size, num_workers, device):
        self.tokenizer = tokenizer
        self.model = model
        self.model.to(device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        
    @property
    @abstractmethod
    def start_token(self):
        pass

    @property
    @abstractmethod
    def end_token(self):
        pass
    
    @property
    def mask_token(self):
        return self.tokenizer.mask_token_id

    def tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer.batch_encode_plus(seqs_str, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]
        try:
            attention_mask = encoded["attention_mask"]
        except:
            attention_mask = None
        if self.start_token is not None:
            starts = torch.where(tokens == self.start_token)[1] + 1 
        else:
            starts = torch.tensor([0]*tokens.shape[0])
        if self.end_token is not None:
            ends = torch.where(tokens == self.end_token)[1]
        else:
            ends = attention_mask.sum(dim=1) 
        return tokens, starts, ends, attention_mask 

    def model_fwd(self, tokens, attention_mask):
        with torch.no_grad():
            try:
                # breakpoint()
                torch_outs = self.model(
                    tokens,
                    attention_mask=attention_mask,
                    encoder_attention_mask=attention_mask
                )
            except:
                torch_outs = self.model(tokens)
            logits = torch_outs.logits.swapaxes(1, 2)
            lls = -F.cross_entropy(logits, tokens, reduction="none")
        return lls

    # @abstractmethod
    # def score(self, tokens, starts, ends, attention_mask):
    #     pass


    def evaluate(self, dataset, output_file, progress_bar=True):
        out_file_obj = open(output_file, "w")
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        for seqs in tqdm(dataloader, disable=(not progress_bar)):
            tokens, starts, ends, attention_mask = self.tokenize(seqs)
            lls = self.score(tokens, starts, ends, attention_mask)
            for lhood in lls.flatten():
                out_file_obj.write(f"{str(lhood)}\n")
                out_file_obj.flush()

class VariantLikelihoodEvaluator(LikelihoodEvaluator):
# dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
#         with h5py.File(out_path + ".tmp", "w") as out_f:
#             allele1_grp = out_f.create_group("allele1")
#             allele2_grp = out_f.create_group("allele2")

#             start = 0
#             for allele1, allele2 in tqdm(dataloader, disable=(not progress_bar)): # shape = batch_size x 500 x 4
#                 if torch.all(allele1 == 0) and torch.all(allele2==0):
#                     continue
#                 end = start + len(allele1)

#                 allele1_tokens, allele1_offsets = self.tokenize(allele1)
#                 allele2_tokens, allele2_offsets = self.tokenize(allele2)

#                 allele1_token_emb = self.model_fwd(allele1_tokens)
#                 allele2_token_emb = self.model_fwd(allele2_tokens)
#                 if self._idx_mode == "variable":
#                     allele1_indices = self._offsets_to_indices(allele1_offsets, allele1)
#                     allele1_indices_dset = allele1_grp.require_dataset("idx_var", (len(dataset), allele1_indices.shape[1]), dtype=np.uint32)
#                     allele1_indices_dset[start:end] = allele1_indices

#                     allele2_indices = self._offsets_to_indices(allele2_offsets, allele2)
#                     allele2_indices_dset = allele2_grp.require_dataset("idx_var", (len(dataset), allele2_indices.shape[1]), dtype=np.uint32)
#                     allele2_indices_dset[start:end] = allele2_indices

#                 elif (start == 0) and (self._idx_mode == "fixed"):
#                     allele1_indices = self._offsets_to_indices(allele1_offsets, allele1)
#                     allele1_indices_dset = allele1_grp.create_dataset("idx_fix", data=allele1_indices, dtype=np.uint32)
#                     allele2_indices = self._offsets_to_indices(allele2_offsets, allele2)
#                     allele2_indices_dset = allele2_grp.create_dataset("idx_fix", data=allele2_indices, dtype=np.uint32)

#                 allele1_grp.create_dataset(f"emb_{start}_{end}", data=allele1_token_emb.numpy(force=True))
#                 allele2_grp.create_dataset(f"emb_{start}_{end}", data=allele2_token_emb.numpy(force=True))

#                 start = end
#         os.rename(out_path + ".tmp", out_path) 


    def evaluate(self, dataset, output_file, progress_bar=True):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        allele1_likelihoods = []
        allele2_likelihoods = []
        
        for allele1, allele2 in tqdm(dataloader, disable=(not progress_bar)):
            tokens_allele1, starts_allele1, ends_allele1, attention_mask_allele1 = self.tokenize(allele1)
            tokens_allele2, starts_allele2, ends_allele2, attention_mask_allele2 = self.tokenize(allele2)
            lls_allele1 = self.score(tokens_allele1, starts_allele1, ends_allele1, attention_mask_allele1)
            lls_allele2 = self.score(tokens_allele2, starts_allele2, ends_allele2, attention_mask_allele2)
            for lhood_allele1, lhood_allele2 in zip(lls_allele1.flatten(), lls_allele2.flatten()):
                allele1_likelihoods.append(lhood_allele1)
                allele2_likelihoods.append(lhood_allele2)
        data = {"allele1_likelihoods" : allele1_likelihoods, "allele2_likelihoods" : allele2_likelihoods}
        df = pl.DataFrame(data, schema={"allele1_likelihoods": pl.Float64, "allele2_likelihoods": pl.Float64})
        df.write_csv(output_file, separator="\t")


class MaskedZeroShotScore(metaclass=ABCMeta):
    @property
    @abstractmethod
    def mask_token(self):
        pass

    def score(self, tokens, starts, ends, attention_mask):
        # breakpoint()
        tokens = tokens.to(device=self.device)
        attention_mask = attention_mask.to(device=self.device)
        lls = torch.zeros(tokens.shape[:2], device=self.device)
        for i in range(tokens.shape[1]):
            clip_mask = ((i >= starts) & (i < ends)).to(device=self.device)
            masked_tokens = tokens.clone()
            masked_tokens[:,i,...] = self.mask_token
            lls[:,i] = self.model_fwd(masked_tokens, attention_mask)[:,i] * clip_mask

        out = lls.sum(dim=1).numpy(force=True)

        return out
    

class CausalZeroShotScore(metaclass=ABCMeta):
    def score(self, tokens, starts, ends, attention_mask):
        tokens = tokens.to(device=self.device)
        lls = self.model_fwd(tokens, attention_mask)
        clip_mask = torch.tensor([[(i >= s) and (i < e) for i in range(lls.shape[1])] for s, e in zip(starts, ends)], 
                                 dtype=torch.float).to(device=self.device)

        out = (lls * clip_mask).sum(1).numpy(force=True)

        return out


class DNABERT2Evaluator(LikelihoodEvaluator, MaskedZeroShotScore):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"zhihan1996/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        config = BertConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_name, config=config, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

    @property
    def start_token(self):
        return 1
    
    @property
    def end_token(self):
        return 2


class GenaLMEvaluator(LikelihoodEvaluator, MaskedZeroShotScore):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"AIRI-Institute/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

    @property
    def start_token(self):
        return 1
    
    @property
    def end_token(self):
        return 2


class HDEvaluator(LikelihoodEvaluator, CausalZeroShotScore):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"LongSafari/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

    @property
    def start_token(self):
        return None
    
    @property
    def end_token(self):
        return 1


class MistralEvaluator(LikelihoodEvaluator, CausalZeroShotScore):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"RaphaelMourad/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

    @property
    def start_token(self):
        return 1
    
    @property
    def end_token(self):
        return 2


class NTEvaluator(LikelihoodEvaluator, MaskedZeroShotScore):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"InstaDeepAI/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

    @property
    def start_token(self):
        return 3
    
    @property
    def end_token(self):
        return None
    
class DNABERT2VariantEvaluator(VariantLikelihoodEvaluator, MaskedZeroShotScore):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"zhihan1996/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        config = BertConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_name, config=config, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

    @property
    def start_token(self):
        return 1
    
    @property
    def end_token(self):
        return 2


class GenaLMVariantEvaluator(VariantLikelihoodEvaluator, MaskedZeroShotScore):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"AIRI-Institute/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

    @property
    def start_token(self):
        return 1
    
    @property
    def end_token(self):
        return 2


class HDVariantEvaluator(VariantLikelihoodEvaluator, CausalZeroShotScore):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"LongSafari/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

    @property
    def start_token(self):
        return None
    
    @property
    def end_token(self):
        return 1


class MistralVariantEvaluator(VariantLikelihoodEvaluator, CausalZeroShotScore):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"RaphaelMourad/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

    @property
    def start_token(self):
        return 1
    
    @property
    def end_token(self):
        return 2


class NTVariantEvaluator(VariantLikelihoodEvaluator, MaskedZeroShotScore):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"InstaDeepAI/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

    @property
    def start_token(self):
        return 3
    
    @property
    def end_token(self):
        return None