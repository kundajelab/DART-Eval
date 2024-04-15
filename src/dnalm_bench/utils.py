import sys
import shutil

import numpy as np

ALPHABET = np.array(["A","C","G","T"], dtype="S1")

def onehot_to_chars(onehot):
	chararray = ALPHABET[np.argmax(onehot, axis=2)]
	strings = [b"".join(row).decode() for row in chararray]

	return strings


def one_hot_encode(sequence):
    sequence = sequence.upper()

    seq_chararray = np.frombuffer(sequence.encode('UTF-8'), dtype='S1')
    one_hot = (seq_chararray[:,None] == ALPHABET[None,:]).astype(np.int8)

    return one_hot


def copy_if_not_exists(src, dst):
    try:
        with open(src, "rb") as sf, open(dst, "xb") as f:
            shutil.copyfileobj(sf, f)
    except FileExistsError:
        pass


class NoModule:
    def __init__(self, *module_names):
        self.module_names = module_names
        self.original_modules = {}

    def __enter__(self):
        for module_name in self.module_names:
            if module_name in sys.modules:
                self.original_modules[module_name] = sys.modules[module_name]
            sys.modules[module_name] = None

    def __exit__(self, exc_type, exc_value, traceback):
        for module_name in self.module_names:
            if module_name in self.original_modules:
                sys.modules[module_name] = self.original_modules[module_name]
            else:
                del sys.modules[module_name]


SEQ_TOKENS = np.array([0, 1, 2, 3], dtype=np.int8)

def dinucleotide_shuffle(seq, rng):
    """
    Adapted from https://github.com/kundajelab/deeplift/blob/0201a218965a263b9dd353099feacbb6f6db0051/deeplift/dinuc_shuffle.py#L43
    """
    tokens = (seq * SEQ_TOKENS[None,:]).sum(axis=1) # Convert one-hot to integer tokens

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

    shuffled = (result[:,None] == SEQ_TOKENS[None,:]).astype(np.int8) # Convert tokens back to one-hot

    return shuffled