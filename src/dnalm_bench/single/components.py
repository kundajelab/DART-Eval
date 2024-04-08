# from abc import ABCMeta, abstractmethod
import hashlib
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import pyfaidx
import pandas as pd
# from scipy.stats import wilcoxon
# from tqdm import tqdm

from ..utils import one_hot_encode, copy_if_not_exists

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
                    copy_if_not_exists(genome_fa, fa_cache_path)
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
                fa_chrom_pos = str(fa[chrom][pos]).upper()
                if fa_chrom_pos==allele1: # allele1 is the reference allele
                        allele2_sequence_data += allele2
                        allele2_sequence_data += str(fa[chrom][pos+1:pos+250].seq)
                elif fa_chrom_pos==allele2:
                        allele1_sequence_data, allele2_sequence_data = allele2_sequence_data, allele1_sequence_data # allele2 is the reference allele
                        allele1_sequence_data += allele1
                        allele1_sequence_data += str(fa[chrom][pos+1:pos+250].seq)
                else: # allele1 and allele2 both do not appear in the reference genome
                        print(chrom, pos, allele1, allele2, " not in reference. In reference, it appears as ", fa_chrom_pos)
                        # still score the SNP by replacing chrom:pos with allele1 and allele2 respectively
                        allele1_sequence_data = str(fa[chrom][pos-250:pos].seq)
                        allele2_sequence_data = str(fa[chrom][pos-250:pos].seq) 
                        allele1_sequence_data += allele1
                        allele2_sequence_data += allele2
                        allele1_sequence_data += str(fa[chrom][pos+1:pos+250].seq)
                        allele2_sequence_data += str(fa[chrom][pos+1:pos+250].seq)

                allele1_sequence = allele1_sequence_data.upper()
                allele2_sequence = allele2_sequence_data.upper()

                fa.close()
                a = start_adj - (pos - 250)
                b = end_adj - (pos - 250)
                allele1_seq[a:b,:] = one_hot_encode(allele1_sequence)
                allele2_seq[a:b,:] = one_hot_encode(allele2_sequence)
                return torch.from_numpy(allele1_seq), torch.from_numpy(allele2_seq)


class FootprintingDataset(Dataset):

        def __init__(self, seqs, seed):
                self.seq_table = pd.read_csv(seqs, sep="\t", header=None)
                self.seed = seed

        def __len__(self):
                return len(self.seq_table)

        def __getitem__(self, idx):
                seq = self.seq_table.loc[idx, 1]
                return torch.from_numpy(one_hot_encode(seq))

