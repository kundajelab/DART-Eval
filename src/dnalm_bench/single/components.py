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
                chrom, start, end, elem_start, elem_end, _, _, rc = self.elements_df.row(idx)

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

class VariantDataset(Dataset):
        _elements_dtypes = {
                "chr": pl.Utf8,
                "pos": pl.UInt32,
                "ref": pl.Utf8,
                "alt": pl.Utf8,
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
                chrom, pos, ref, alt = self.elements_df.row(idx)

                # Extract the sequence
                window = 500
                ref_seq = np.zeros((window, 4), dtype=np.int8)
                alt_seq = np.zeros((window, 4), dtype=np.int8)
                fa = pyfaidx.Fasta(self.genome_fa, one_based_attributes=False)

                # extend the sequence by -249 on the left of pos and +250 on the right of pos
                sequence_data = fa[chrom][max(0, pos-249), pos+250] # check if pos+250 goes outside chrom
                sequence = sequence_data.seq.upper()
                start_adj = sequence_data.start
                end_adj = sequence_data.end
                alt_sequence_data = fa[chrom][max(0, pos-249):pos]
                alt_sequence_data += alt
                alt_sequence_data += fa[chrom][pos+1:pos+250]
                alt_sequence = alt_sequence_data.upper()
                
                fa.close()

                a = start_adj - start
                b = end_adj - start
                ref_seq[:end_adj,:] = one_hot_encode(sequence)
                alt_seq[:end_adj,:] = one_hot_encode(alt_sequence)
                return torch.from_numpy(ref_seq), torch.from_numpy(alt_seq)
