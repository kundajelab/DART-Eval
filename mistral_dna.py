import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForCausalLM

import components

class MistralEvaluator(components.CausalLogPerplexityEvaluator):
	def __init__(self, model_name, genome_fa, elements_tsv, chroms, batch_size, num_workers, seed, device):
		super().__init__(genome_fa, elements_tsv, chroms, batch_size, num_workers, seed, device)
		self.model_name = f"RaphaelMourad/{model_name}"
		self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
		self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
		self.model.to(device)

	def tokenize(self, seqs):
		seqs_str = components.onehot_to_chars(seqs)
		encoded = self.tokenizer.batch_encode_plus(seqs_str, return_tensors="pt", padding=True)
		tokens = encoded["input_ids"]
		starts, ends = torch.where(tokens == 1)[1] + 1, torch.where(tokens == 2)[1] - 1
		return tokens, starts, ends, encoded["attention_mask"] 


	def model_fwd(self, tokens):
		with torch.no_grad():
			logits = self.model(tokens)["logits"]
			return F.cross_entropy(logits.swapaxes(1, 2), tokens, reduction="none")





def main():
	model_name = "Mistral-DNA-v0.1"
	genome_fa = "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
	elements_tsv = "~/scratch/jitter50.bed"
	chroms = ["chr1"]
	batch_size = 1024
	num_workers = 4
	seed = 0
	device = "cuda"
	
	evaluator = MistralEvaluator(model_name, genome_fa, elements_tsv, chroms, batch_size, num_workers, seed, device)
	acc, pval, signed_rank_sum = evaluator.evaluate(progress_bar=True)

	print(f"Accuracy: {acc}")
	print(f"P-value: {pval}")
	print(f"Signed Rank Sum: {signed_rank_sum}")


if __name__ == "__main__":
	main()




