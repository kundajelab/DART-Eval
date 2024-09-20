import pandas as pd
import h5py
import numpy as np
import os

work_dir = os.environ.get("DART_WORK_DIR", "")

ALPHABET = np.array(["A","C","G","T"], dtype="S1")
def one_hot_encode(sequence):
    sequence = sequence.upper()

    seq_chararray = np.frombuffer(sequence.encode('UTF-8'), dtype='S1')
    one_hot = (seq_chararray[:,None] == ALPHABET[None,:]).astype(np.int8)

    return one_hot

if __name__ == "__main__":
    dataset_file = os.path.join(work_dir, "task_2_footprinting/processed_data/footprint_dataset_350_v1.txt")
    seq_data = pd.read_csv(dataset_file, sep="\t", header=None)

    seq_data["motif"] = seq_data[0].apply(lambda x: x.split("_")[1])
    seq_data["motif"] = seq_data[0].apply(lambda x: x.split("_")[1])
    seq_data["type"] = seq_data[0].apply(lambda x: x.split("_")[-1])

    raw_string_seqs = seq_data.loc[seq_data["motif"] == "raw"][1].values

    true_motif_string_seqs, shuff_motif_string_seqs = [], []
    for motif in np.unique(seq_data["motif"]):
        if motif == "raw":
            continue
        true_seqs = seq_data.loc[(seq_data["motif"] == motif) & (seq_data["type"] == "true")]
        shuf_seqs = seq_data.loc[(seq_data["motif"] == motif) & (seq_data["type"] == "shuffled")]
        true_motif_string_seqs.append(true_seqs[1].values)
        shuff_motif_string_seqs.append(shuf_seqs[1].values)

    raw_seqs = np.expand_dims(np.array([one_hot_encode(x) for x in raw_string_seqs]), 0)
    true_motif_seqs = np.array([[one_hot_encode(x) for x in y] for y in true_motif_string_seqs])
    shuff_motif_seqs = np.array([[one_hot_encode(x) for x in y] for y in shuff_motif_string_seqs])

    out_path = os.path.join(work_dir, "task_2_footprinting/data.h5")

    with h5py.File(out_path, "w") as f:
        f.create_dataset(name="raw", data=raw_seqs, dtype=np.uint8, compression="gzip")
        f.create_dataset(name="positive", data=true_motif_seqs, dtype=np.uint8, compression="gzip")
        f.create_dataset(name="shuffled", data=shuff_motif_seqs, dtype=np.uint8, compression="gzip")

