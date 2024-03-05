# from abc import ABCMeta, abstractmethod
import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import pyfaidx
# from scipy.stats import wilcoxon
# from tqdm import tqdm

from ..utils import one_hot_encode

class SimpleSequence(Dataset):
	_elements_dtypes = {
		"chr": pl.Utf8,
		"input_start": pl.UInt32,
		"input_end": pl.UInt32,
		"elem_start": pl.UInt32,
		"elem_end": pl.UInt32,
	}

	_seq_tokens = np.array([0, 1, 2, 3], dtype=np.int8)

	_seed_upper = 2**128

	def __init__(self, genome_fa, elements_tsv, chroms, seed):
		super().__init__()

		self.seed = seed

		self.elements_df = self._load_elements(elements_tsv, chroms)

		self.genome_fa = genome_fa
		fa = pyfaidx.Fasta(self.genome_fa) # Build index if needed
		fa.close()

	@classmethod
	def _load_elements(cls, elements_file, chroms):
		df = pl.scan_csv(elements_file, separator="\t", quote_char=None, dtypes=cls._elements_dtypes)
		
		if chroms is not None:
			df = df.filter(pl.col("chr").is_in(chroms))

		df = df.collect()

		return df

	
	def __len__(self):
		return self.elements_df.height
	
	def __getitem__(self, idx):
		chrom, start, end, elem_start, elem_end, _, _ = self.elements_df.row(idx)

		# Extract the sequence
		window = end - start
		seq = np.zeros((window, 4), dtype=np.int8)

		fa = pyfaidx.Fasta(self.genome_fa, one_based_attributes=False)

		sequence_data = fa[chrom][max(0, start):end]
		sequence = sequence_data.seq.upper()
		start_adj = sequence_data.start
		end_adj = sequence_data.end

		fa.close()

		a = start_adj - start
		b = end_adj - start
		seq[a:b,:] = one_hot_encode(sequence)

		return torch.from_numpy(seq)
	

class VariantDataLoader(DataLoader):
	def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
		 batch_sampler=None, num_workers=0, collate_fn=None,
		 pin_memory=False, drop_last=False, timeout=0,
		 worker_init_fn=None, dummy=None):
		self.dataset = dataset
		super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
				num_workers, collate_fn, pin_memory, drop_last, timeout,
				worker_init_fn, dummy)
		
	
	def __iter__(self):
		for batch in super().__iter__():
		# Filter out None items in each batch
			filtered_batch = [item for item in batch if item is not None]
			yield filtered_batch