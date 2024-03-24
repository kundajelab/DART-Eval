import sys

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


def loop_infinite(iterable):
    while True:
        yield from iterable


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


