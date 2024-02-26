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