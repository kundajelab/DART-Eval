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
                chrom, pos, allele1, allele2 = self.elements_df.row(idx)[:4]

                # 1-indexed position
                pos = int(pos) - 1
                # Extract the sequence
                window = 500
                allele1_seq = np.zeros((window, 4), dtype=np.int8)
                allele2_seq = np.zeros((window, 4), dtype=np.int8)
                fa = pyfaidx.Fasta(self.genome_fa, one_based_attributes=False)

                # extend the sequence by -249 on the left of pos and +250 on the right of pos
                allele1_sequence_data = fa[chrom][pos-250:pos+250] # check if pos+250 goes outside chrom
                start_adj = allele1_sequence_data.start
                end_adj = allele1_sequence_data.end
                allele1_sequence_data = str(allele1_sequence_data.seq)
                allele2_sequence_data = str(fa[chrom][pos-250:pos].seq)
                if fa[chrom][pos]==allele1: # allele1 is the reference allele
                        allele2_sequence_data += allele2
                        allele2_sequence_data += str(fa[chrom][pos+1:pos+250].seq)
                elif fa[chrom][pos]==allele2:
                        allele1_sequence_data, allele2_sequence_data = allele2_sequence_data, allele1_sequence_data # allele2 is the reference allele
                        allele1_sequence_data += allele1
                        allele1_sequence_data += str(fa[chrom][pos+1:pos+250].seq)
                else: # allele1 and allele2 both do not appear in the reference genome
                        print(chrom, pos, allele1, allele2, " not in reference.")
                        return torch.from_numpy(allele1_seq), torch.from_numpy(allele2_seq)

                # print(allele1_sequence_data, allele2_sequence_data)
                allele1_sequence = allele1_sequence_data.upper()
                allele2_sequence = allele2_sequence_data.upper()
                

                fa.close()
                a = start_adj - (pos - 250)
                b = end_adj - (pos - 250)
                allele1_seq[a:b,:] = one_hot_encode(allele1_sequence)
                allele2_seq[a:b,:] = one_hot_encode(allele2_sequence)
                return torch.from_numpy(allele1_seq), torch.from_numpy(allele2_seq)
