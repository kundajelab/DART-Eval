from abc import ABCMeta, abstractmethod
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import pyfaidx
from scipy.stats import wilcoxon
from tqdm import tqdm

class ElementsDataset(Dataset):
    # _bed_schema = ["chr", "start", "end"]
    # _bed_dtypes = [pl.Utf8, pl.UInt32, pl.UInt32]

    _elements_dtypes = {
        "chr": pl.Utf8,
        "input_start": pl.UInt32,
        "input_end": pl.UInt32,
        "ccre_start": pl.UInt32,
        "ccre_end": pl.UInt32,
        "ccre_relative_start": pl.UInt32,
        "ccre_relative_end": pl.UInt32,
        "reverse_complement": pl.Boolean
    }

    _seq_alphabet = np.array(["A","C","G","T"], dtype="S1")
    _seq_tokens = np.array([0, 1, 2, 3], dtype=np.int8)

    _seed_upper = 2**128

    def __init__(self, genome_fa, elements_tsv, chroms, seed):
        super().__init__()

        # if window % 2 != 0:
        #     raise ValueError(f"Window size {window} must be even.")

        # self.window = window
        # self.max_jitter = max_jitter
        # self.reverse_complement = reverse_complement
        self.seed = seed

        # self.elements_df = self._load_bed(elements_bed, chroms)
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
    
    # @classmethod
    # def _load_bed(cls, bed_file, chroms):
    #     df = (
    #         pl.scan_csv(bed_file, has_header=False, separator="\t", quote_char=None,
    #                     new_columns=cls._bed_schema, dtypes=cls._bed_dtypes)
    #         .select(cls._bed_schema)
    #     )

    #     if chroms is not None:
    #         df = df.filter(pl.col("chr").is_in(chroms))

    #     df = df.collect()

    #     return df

    @classmethod
    def _one_hot_encode(cls, sequence):
        sequence = sequence.upper()

        seq_chararray = np.frombuffer(sequence.encode('UTF-8'), dtype='S1')
        one_hot = (seq_chararray[:,None] == cls._seq_alphabet[None,:]).astype(np.int8)

        return one_hot

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
        # chrom, elem_start, elem_end = self.elements_df.row(idx)
        # center = elem_start + (elem_end - elem_start) // 2
        # start = center - self.window // 2
        # end = start + self.window

        chrom, start, end, elem_start, elem_end, _, _, rc = self.elements_df.row(idx)

        item_seed = hash((self.seed, chrom, elem_start, elem_end),) % self._seed_upper
        # print(sys.getsizeof(item_seed)) ####
        rng = np.random.default_rng(item_seed)

        # # Jitter the region
        # jitter = rng.integers(-self.max_jitter, self.max_jitter, endpoint=True)
        # start += jitter
        # end += jitter

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
        seq[a:b,:] = self._one_hot_encode(sequence)

        # Generate shuffled control
        e_a = max(elem_start - start, a)
        e_b = min(elem_end - start, b)
        elem = seq[e_a:e_b,:]
        shuf = self._dinuc_shuffle(elem, rng)
        ctrl = seq.copy()
        ctrl[e_a:e_b,:] = shuf
        
        # Reverse complement augment
        # if self.reverse_complement and rng.choice([True, False]):
        if rc:
            seq = seq[::-1,::-1].copy()
            ctrl = ctrl[::-1,::-1].copy()

        return torch.from_numpy(seq), torch.from_numpy(ctrl)


def onehot_to_chars(onehot):
    alphabet = np.array(["A","C","G","T"], dtype="S1")
    chararray = alphabet[np.argmax(onehot, axis=2)]
    strings = [b"".join(row).decode() for row in chararray]

    return strings


class Evaluator(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, genome_fa, elements_tsv, chroms, batch_size, num_workers, seed, device):
        self.dataset = ElementsDataset(genome_fa, elements_tsv, chroms, seed)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.device = device

    @abstractmethod
    def tokenize(self, seqs):
        pass

    @abstractmethod
    def model_fwd(self, tokens, attention_mask):
        pass

    @abstractmethod
    def score(self, tokens, starts, ends, attention_mask):
        pass
    
    def evaluate(self, progress_bar=False):
        diffs_lst = []
        corrects_lst = []

        for seqs, ctrls in tqdm(self.dataloader, disable=(not progress_bar)):
            seq_tokens, seq_starts, seq_ends, seq_attention_mask = self.tokenize(seqs)
            ctrl_tokens, ctrl_starts, ctrl_ends, ctrl_attention_mask = self.tokenize(ctrls)

            seq_scores = self.score(seq_tokens, seq_starts, seq_ends, seq_attention_mask)
            ctrl_scores = self.score(ctrl_tokens, ctrl_starts, ctrl_ends, ctrl_attention_mask)

            diff_batch = seq_scores - ctrl_scores
            correct_batch = diff_batch > 0

            diffs_lst.append(diff_batch)
            corrects_lst.append(correct_batch)

        diffs = np.concatenate(diffs_lst)
        corrects = np.concatenate(corrects_lst)

        acc = corrects.mean()

        wilcox = wilcoxon(diffs, alternative="greater")
        pval = wilcox.pvalue
        signed_rank_sum = wilcox.statistic

        return acc, pval, signed_rank_sum


class MaskedLogPerplexityEvaluator(Evaluator, metaclass=ABCMeta):
    def score(self, tokens, starts, ends, attention_mask):
        tokens = tokens.to(device=self.device)
        attention_mask = attention_mask.to(device=self.device)
        lls = torch.zeros(tokens.shape[:2], device=self.device)
        for i in range(tokens.shape[1]):
            clip_mask = ((i >= starts) & (i < ends)).to(device=self.device)
            masked_tokens = tokens.clone()
            masked_tokens[:,i,...] = self.mask_token
            lls[:,i] = self.model_fwd(masked_tokens, attention_mask)[:,i] * clip_mask

        lp = lls.sum(dim=1).numpy(force=True)

        return lp
    



