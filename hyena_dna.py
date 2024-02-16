import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


import components

class HDEvaluator(components.CausalLogPerplexityEvaluator):
    def __init__(self, model_name, genome_fa, elements_tsv, chroms, batch_size, num_workers, seed, device):
        super().__init__(genome_fa, elements_tsv, chroms, batch_size, num_workers, seed, device)
        self.model_name = f"LongSafari/{model_name}"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, padding_side="right")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.to(device)

    @property
    def mask_token(self):
        return self.tokenizer.mask_token_id

    def tokenize(self, seqs):
        seqs_str = components.onehot_to_chars(seqs)
        encoded = self.tokenizer.batch_encode_plus(seqs_str, return_tensors="pt", padding=True, return_attention_mask=True)
        tokens = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        starts = torch.tensor([0]*tokens.shape[0])
        ends = torch.where(tokens == 1)[1]
        return tokens, starts, ends, attention_mask
    
    def model_fwd(self, tokens):
        with torch.no_grad():
            torch_outs = self.model(tokens)
            logits = torch_outs.logits.swapaxes(1,2)
            lls = -F.cross_entropy(logits, tokens, reduction="none")
        torch.cuda.synchronize()
        return lls


if __name__ == "__main__":
    model_name = "hyenadna-large-1m-seqlen-hf"
    genome_fa = "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    elements_tsv = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/regions/ccre_test_regions_500_jitter_50.bed"
    chroms = ["chr22"]
    batch_size = 1024
    num_workers = 4
    seed = 0
    device = "cuda"
    
    evaluator = HDEvaluator(model_name, genome_fa, elements_tsv, chroms, batch_size, num_workers, seed, device)
    metrics = evaluator.evaluate(progress_bar=True)

    for k, v in metrics.items():
        print(f"{k}: {v}")
