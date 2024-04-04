from abc import ABCMeta, abstractmethod
import os
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForCausalLM, BertConfig
from scipy.stats import wilcoxon
from tqdm import tqdm

from ..components import PairedControlDataset
from ...utils import onehot_to_chars

class MaskedZeroShotScore(metaclass=ABCMeta):
    @property
    @abstractmethod
    def mask_token(self):
        pass

    def score(self, tokens, starts, ends, attention_mask):
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
        attention_mask = attention_mask.to(device=self.device)
        lls = self.model_fwd(tokens, attention_mask)
        clip_mask = torch.tensor([[(i >= s) and (i < e) for i in range(lls.shape[1])] for s, e in zip(starts, ends)], 
                                 dtype=torch.float).to(device=self.device)

        out = (lls * clip_mask).sum(1).numpy(force=True)

        return out


class ZeroShotPairedControlEvaluator(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, dataset, batch_size, num_workers, device):
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.device = device

    @abstractmethod
    def tokenize(self, seqs):
        pass

    @abstractmethod
    def model_fwd(self, tokens, attention_mask):
        pass

    # @abstractmethod
    # def score(self, tokens, starts, ends, attention_mask):
    #     pass
    
    def evaluate(self, out_dir, progress_bar=False):
        os.makedirs(out_dir, exist_ok=True)
        scores_path = os.path.join(out_dir, "scores.tsv")
        metrics_path = os.path.join(out_dir, "metrics.json")

        with open(scores_path, "w") as f:
            f.write("idx\tseq_score\tctrl_score\n")

            metrics = {}
            diffs_lst = []
            corrects_lst = []
            
            for seqs, ctrls, inds in tqdm(self.dataloader, disable=(not progress_bar), ncols=120):
                seq_tokens, seq_starts, seq_ends, seq_attention_mask = self.tokenize(seqs)
                ctrl_tokens, ctrl_starts, ctrl_ends, ctrl_attention_mask = self.tokenize(ctrls)

                seq_scores = self.score(seq_tokens, seq_starts, seq_ends, seq_attention_mask)
                ctrl_scores = self.score(ctrl_tokens, ctrl_starts, ctrl_ends, ctrl_attention_mask)

                for seq_score, ctrl_score in zip(seq_scores, ctrl_scores):
                    f.write(f"{inds}\t{seq_score}\t{ctrl_score}\n")
                f.flush()

                diff_batch = seq_scores - ctrl_scores
                correct_batch = diff_batch > 0

                diffs_lst.append(diff_batch)
                corrects_lst.append(correct_batch)

            diffs = np.concatenate(diffs_lst)
            corrects = np.concatenate(corrects_lst)

        metrics["acc"] = corrects.mean()

        wilcox = wilcoxon(diffs, alternative="greater")
        metrics["pval"] = wilcox.pvalue
        metrics["signed_rank_sum"] = wilcox.statistic
        metrics["mean_diff"] = diffs.mean()
        metrics["q05_diff"] = np.percentile(diffs, 5)
        metrics["q25_diff"] = np.percentile(diffs, 25)
        metrics["median_diff"] = np.median(diffs)
        metrics["q75_diff"] = np.percentile(diffs, 75)
        metrics["q95_diff"] = np.percentile(diffs, 95)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        return metrics


class HFZeroShotEvaluator(ZeroShotPairedControlEvaluator, metaclass=ABCMeta):
    def __init__(self, tokenizer, model, dataset, batch_size, num_workers, device):
        self.tokenizer = tokenizer
        self.model = model
        self.model.to(device)
        super().__init__(dataset, batch_size, num_workers, device)

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
        attention_mask = encoded.get("attention_mask")
        if self.start_token is not None:
            starts = torch.where(tokens == self.start_token)[1] + 1 
        else:
            starts = 0
        if self.end_token is not None:
            ends = torch.where(tokens == self.end_token)[1]
        else:
            ends = attention_mask.sum(dim=1) 
        return tokens, starts, ends, attention_mask 

    def model_fwd(self, tokens, attention_mask):
        with torch.no_grad():
            torch_outs = self.model(
                tokens,
                attention_mask=attention_mask,
                encoder_attention_mask=attention_mask
            )
            logits = torch_outs.logits.swapaxes(1, 2)
            lls = -F.cross_entropy(logits, tokens, reduction="none")
        return lls
    

class DNABERT2Evaluator(HFZeroShotEvaluator, MaskedZeroShotScore):
    def __init__(self, model_name, dataset, batch_size, num_workers, device):
        model_name = f"zhihan1996/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        config = BertConfig.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_name, config=config, trust_remote_code=True)
        super().__init__(tokenizer, model, dataset, batch_size, num_workers, device)

    @property
    def start_token(self):
        return 1
    
    @property
    def end_token(self):
        return 2


class GenaLMEvaluator(HFZeroShotEvaluator, MaskedZeroShotScore):
    def __init__(self, model_name, dataset, batch_size, num_workers, device):
        model_name = f"AIRI-Institute/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, dataset, batch_size, num_workers, device)

    @property
    def start_token(self):
        return 1
    
    @property
    def end_token(self):
        return 2


class HDEvaluator(HFZeroShotEvaluator, MaskedZeroShotScore):
    def __init__(self, model_name, dataset, batch_size, num_workers, device):
        model_name = f"LongSafari/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, dataset, batch_size, num_workers, device)

    @property
    def start_token(self):
        return None
    
    @property
    def end_token(self):
        return 1

    def model_fwd(self, tokens, attention_mask):
        with torch.no_grad():
            torch_outs = self.model(
                tokens
            )
            logits = torch_outs.logits.swapaxes(1, 2)
            lls = -F.cross_entropy(logits, tokens, reduction="none")
        return lls


class MistralEvaluator(HFZeroShotEvaluator, CausalZeroShotScore):
    def __init__(self, model_name, dataset, batch_size, num_workers, device):
        model_name = f"RaphaelMourad/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, dataset, batch_size, num_workers, device)

    @property
    def start_token(self):
        return 1
    
    @property
    def end_token(self):
        return 2


class NTEvaluator(HFZeroShotEvaluator, MaskedZeroShotScore):
    def __init__(self, model_name, dataset, batch_size, num_workers, device):
        super().__init__(dataset, batch_size, num_workers, device)
        model_name = f"InstaDeepAI/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, dataset, batch_size, num_workers, device)

    @property
    def start_token(self):
        return 3
    
    @property
    def end_token(self):
        return None