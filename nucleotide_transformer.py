import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForMaskedLM

import components


class NTEvaluator(components.MaskedLogPerplexityEvaluator):
    def __init__(self, model_name, genome_fa, elements_tsv, chroms, batch_size, num_workers, seed, device):
        super().__init__(genome_fa, elements_tsv, chroms, batch_size, num_workers, seed, device)
        self.model_name = f"InstaDeepAI/{model_name}"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.to(device)
        # print(self.model) ####

    @property
    def mask_token(self):
        return self.tokenizer.mask_token_id

    def tokenize(self, seqs):
        seqs_str = components.onehot_to_chars(seqs)
        encoded = self.tokenizer.batch_encode_plus(seqs_str, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        starts = 1
        ends = attention_mask.sum(dim=1) - 1

        return tokens, starts, ends, attention_mask
    
    def model_fwd(self, tokens, attention_mask):
        # print(torch.cuda.memory_allocated(), torch.cuda.memory_cached()) ####
        # print(tokens) ####
        # a = time.perf_counter() ####
        # with profile(activities=[
        # ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof: ####
        #     with record_function("model_inference"): ####
        with torch.no_grad():
            torch_outs = self.model(
                tokens,
                attention_mask=attention_mask,
                encoder_attention_mask=attention_mask
            )
            logits = torch_outs.logits.swapaxes(1, 2)
            lls = -F.cross_entropy(logits, tokens, reduction="none")
        # print(lls) ####
        # torch.cuda.synchronize() ####
        # print(time.perf_counter() - a) ####
                    
        # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)) ####


        return lls


if __name__ == "__main__":
    # model_name = "nucleotide-transformer-500m-human-ref"
    model_name = "nucleotide-transformer-v2-500m-multi-species"
    # genome_fa = "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    genome_fa = "/mnt/data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    # elements_tsv = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/regions/ccre_test_regions.bed"
    elements_tsv = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/regions/ccre_test_regions_500_jitter_50.bed"
    chroms = ["chr22"]
    batch_size = 1024
    # batch_size = 1 ####
    num_workers = 4
    seed = 0
    device = "cuda"
    
    evaluator = NTEvaluator(model_name, genome_fa, elements_tsv, chroms, batch_size, num_workers, seed, device)
    acc, pval, signed_rank_sum, mean_diff = evaluator.evaluate(progress_bar=True)

    print(f"Accuracy: {acc}")
    print(f"P-value: {pval}")
    print(f"Signed Rank Sum: {signed_rank_sum}")
    print(f"Mean Score Difference: {mean_diff}")