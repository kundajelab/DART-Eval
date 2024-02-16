from abc import ABCMeta, abstractmethod
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForCausalLM, BertConfig, BertForMaskedLM
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
import torch.nn.functional as F
from gpn.data import GenomeMSA, Tokenizer
import gpn.model
import numpy as np
import components
import hashlib
from scipy.stats import wilcoxon


class GPNMSADataset(components.ElementsDataset):
	def __init__(self, genome_fa, msa, elements_tsv, chroms, seed):
		super().__init__(genome_fa, elements_tsv, chroms, seed)
		_seq_alphabet = np.array(["-", "A","C","G","T"], dtype="S1")
		_seq_tokens = np.array([0, 1, 2, 3, 4], dtype=np.int8)
		self.msa = msa

	@classmethod
	def _dinuc_shuffle(cls, tokens, rng):
		"""
		Adapted from https://github.com/kundajelab/deeplift/blob/0201a218965a263b9dd353099feacbb6f6db0051/deeplift/dinuc_shuffle.py#L43
		"""
		new_ind_order = []

		# For each token, get a list of indices of all the tokens that come after it
		shuf_next_inds = []
		for t in range(5):
			mask = tokens[:-1] == t  # Excluding last char
			inds = np.where(mask)[0]
			shuf_next_inds.append(inds + 1)  # Add 1 for next token

		# Shuffle the next indices
		for t in range(5):
			inds = np.arange(len(shuf_next_inds[t]))
			inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
			shuf_next_inds[t] = shuf_next_inds[t][inds]

		counters = [0, 0, 0, 0, 0]
	
		# Build the resulting array
		ind = 0
		result = np.empty_like(tokens)
		result[0] = tokens[ind]
		new_ind_order.append(ind)
		for j in range(1, len(tokens)):
			t = tokens[ind]
			ind = shuf_next_inds[t][counters[t]]
			counters[t] += 1
			result[j] = tokens[ind]
			new_ind_order.append(ind)


		return result, np.array(new_ind_order)

	def __len__(self):
		print("Calculating length 1")
		return self.elements_df.height


	def __getitem__(self, idx):
		#This works a little differently than the other datasets
		# We are going to dinucleotide shuffle the tokenized sequences extracted from the MSA
		#We will also return the tokenized sequence rather than the raw sequence
		chrom, start, end, elem_start, elem_end, elem_rel_start, elem_rel_end, rc = self.elements_df.row(idx)
		item_bytes = (self.seed, chrom, elem_start, elem_end).__repr__().encode('utf-8')
		item_seed = int(hashlib.sha256(item_bytes).hexdigest(), 16) % self._seed_upper
		rng = np.random.default_rng(item_seed)
		msa_tokens = self.msa.get_msa(chrom[3:], start, end, strand="+", tokenize=True)
		elem_human = msa_tokens[elem_rel_start:elem_rel_end,0]
		_, shuf_inds = self._dinuc_shuffle(elem_human, rng)
		control_tokens = np.concatenate([msa_tokens[:elem_rel_start], msa_tokens[shuf_inds + elem_rel_start], msa_tokens[elem_rel_end:]])
		if rc:
			reversed_seq, reversed_control = 5 - msa_tokens[::-1], 5 - control_tokens[::-1]
			msa_tokens, control_tokens = np.where(reversed_seq == 5, 0, reversed_seq), np.where(reversed_control == 5, 0, reversed_control)

		assert msa_tokens.sum() == control_tokens.sum()

		return torch.from_numpy(msa_tokens), torch.from_numpy(control_tokens)







class GPNMSAEvaluator(components.MaskedLogPerplexityEvaluator):
		def __init__(self, model_name, genome_fa, elements_tsv, chroms, batch_size, num_workers, seed, device):
			super().__init__(genome_fa, elements_tsv, chroms, batch_size, num_workers, seed, device)
			self.model_name = f"songlab/{model_name}"
			msa_path = "zip:///::https://huggingface.co/datasets/songlab/multiz100way/resolve/main/89.zarr.zip"
			self.genome_msa = GenomeMSA(msa_path)
			self.dataset = GPNMSADataset(genome_fa, self.genome_msa, elements_tsv, chroms, seed)
			self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
			self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, trust_remote_code=True)
			self.model.to(device)
			self.tokenizer = Tokenizer()

		@property
		def mask_token(self):
			return self.tokenizer.mask_token_id()


		def tokenize(self, seqs):
			start, end = torch.tensor([0] * len(seqs)), torch.tensor([seqs.shape[1]] * len(seqs)) #assuming no padding
			return seqs, start, end, None 

		def score(self, tokens, starts, ends, attention_mask):
			tokens, aux_features = tokens[:,:,0], tokens[:,:,1:]
			tokens = tokens.to(device=self.device)
			aux_features = aux_features.to(self.device)
			lls = torch.zeros(tokens.shape[:2], device=self.device)
			for i in range(tokens.shape[1]):
				clip_mask = ((i >= starts) & (i < ends)).to(device=self.device)
				masked_tokens = tokens.clone()
				masked_tokens[:,i,...] = self.mask_token
				lls[:,i] = self.model_fwd(masked_tokens.long(), aux_features.long())[:,i] * clip_mask

			lp = lls.sum(dim=1).numpy(force=True)

			return lp


		def model_fwd(self, tokens, aux_features):
			with torch.no_grad():
				torch_outs = self.model(
					input_ids=tokens,
					aux_features=aux_features
				)
				logits = torch_outs.logits.swapaxes(1, 2)
				lls = -F.cross_entropy(logits, tokens, reduction="none")
			return lls


if __name__ == "__main__":
	model_name = "gpn-msa-sapiens"
	genome_fa = "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
	elements_tsv = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/regions/ccre_test_regions_500_jitter_50.bed"
	chroms = ["chr22"]
	batch_size = 64
	num_workers = 4
	seed = 0
	device = "cuda"
	
	evaluator = GPNMSAEvaluator(model_name, genome_fa, elements_tsv, chroms, batch_size, num_workers, seed, device)
	acc, pval, signed_rank_sum = evaluator.evaluate(progress_bar=True)

	print(f"Accuracy: {acc}")
	print(f"P-value: {pval}")
	print(f"Signed Rank Sum: {signed_rank_sum}")


