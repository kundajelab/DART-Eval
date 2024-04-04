# from abc import ABCMeta, abstractmethod
import hashlib
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import pyfaidx
# from scipy.stats import wilcoxon
# from tqdm import tqdm

from ..utils import one_hot_encode

class PairedControlDataset(Dataset):
    _elements_dtypes = {
        "chr": pl.Utf8,
        "input_start": pl.UInt32,
        "input_end": pl.UInt32,
        "ccre_start": pl.UInt32,
        "ccre_end": pl.UInt32,
        "ccre_relative_start": pl.Int32,
        "ccre_relative_end": pl.Int32,
        "reverse_complement": pl.Boolean
    }

    _seq_tokens = np.array([0, 1, 2, 3], dtype=np.int8)

    _seed_upper = 2**128

    def __init__(self, genome_fa, elements_tsv, chroms, seed, cache_dir=None):
        super().__init__()

        self.seed = seed

        self.elements_df = self._load_elements(elements_tsv, chroms)

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

            fa_path_abs = os.path.abspath(genome_fa)
            fa_idx_path_abs = fa_path_abs + ".fai"
            fa_path_hash = hashlib.sha256(fa_path_abs.encode('utf-8')).hexdigest()
            fa_cache_path = os.path.join(cache_dir, fa_path_hash + ".fa")
            fa_idx_cache_path = fa_cache_path + ".fai"
            self._copy_if_not_exists(genome_fa, fa_cache_path)
            genome_fa = fa_cache_path
            try:
                self._copy_if_not_exists(fa_idx_path_abs, fa_idx_cache_path)
            except FileNotFoundError:
                pass

        self.genome_fa = genome_fa
        fa = pyfaidx.Fasta(self.genome_fa) # Build index if needed
        fa.close()

    @classmethod
    def _load_elements(cls, elements_file, chroms):
        df = pl.scan_csv(elements_file, separator="\t", quote_char=None, dtypes=cls._elements_dtypes).with_row_index()
        
        if chroms is not None:
            df = df.filter(pl.col("chr").is_in(chroms))

        df = df.collect()

        return df

    @classmethod
    def _dinuc_shuffle(cls, seq, rng):
        """
        Adapted from https://github.com/kundajelab/deeplift/blob/0201a218965a263b9dd353099feacbb6f6db0051/deeplift/dinuc_shuffle.py#L43
        """
        tokens = (seq * cls._seq_tokens[None,:]).sum(axis=1) # Convert one-hot to integer tokens

        # For each token, get a list of indices of all the tokens that come after it
        shuf_next_inds = []
        for t in range(4):
            mask = tokens[:-1] == t  # Excluding last char
            inds = np.where(mask)[0]
            shuf_next_inds.append(inds + 1)  # Add 1 for next token

        # Shuffle the next indices
        for t in range(4):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0, 0, 0, 0]
    
        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        shuffled = (result[:,None] == cls._seq_tokens[None,:]).astype(np.int8) # Convert tokens back to one-hot

        return shuffled
    
    def __len__(self):
        return self.elements_df.height
    
    def __getitem__(self, idx):
        idx_orig, chrom, start, end, elem_start, elem_end, _, _, rc = self.elements_df.row(idx)

        item_bytes = (self.seed, chrom, elem_start, elem_end).__repr__().encode('utf-8')
        item_seed = int(hashlib.sha256(item_bytes).hexdigest(), 16) % self._seed_upper
        
        rng = np.random.default_rng(item_seed)

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

        # Generate shuffled control
        e_a = max(elem_start - start, a)
        e_b = min(elem_end - start, b)
        elem = seq[e_a:e_b,:]
        shuf = self._dinuc_shuffle(elem, rng)
        ctrl = seq.copy()
        ctrl[e_a:e_b,:] = shuf
        
        # Reverse complement augment
        if rc:
            seq = seq[::-1,::-1].copy()
            ctrl = ctrl[::-1,::-1].copy()

        return torch.from_numpy(seq), torch.from_numpy(ctrl), torch.tensor(idx_orig)
