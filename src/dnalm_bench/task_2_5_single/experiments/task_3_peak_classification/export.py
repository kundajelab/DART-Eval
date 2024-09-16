import os

import numpy as np
import h5py
from torch.utils.data import DataLoader

from ...finetune import PeaksEndToEndDataset

work_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    genome_fa = os.path.join(work_dir, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    elements_tsv = os.path.join(work_dir,"task_3_peak_classification/processed_inputs/peaks_by_cell_label_unique_dataloader_format.tsv")

    chroms_train = [
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr7",
        "chr8",
        "chr9",
        "chr11",
        "chr12",
        "chr13",
        "chr15",
        "chr16",
        "chr17",
        "chr19",
        "chrX",
        "chrY"
    ]
    
    chroms_val = [
        "chr6",
        "chr21"
    ]

    chroms_test = [
        "chr5",
        "chr10",
        "chr14",
        "chr18",
        "chr20",
        "chr22"
    ]

    classes = {
        "GM12878": 0,
        "H1ESC": 1,
        "HEPG2": 2,
        "IMR90": 3,
        "K562": 4
    } 

    out_path = os.path.join(work_dir, f"task_3_peak_classification/data.h5")

    batch_size = 8192

    with h5py.File(out_path, "w") as f:
        for mode, chroms in zip(["train", "val", "test"], [chroms_train, chroms_val, chroms_test]):
            dataset = PeaksEndToEndDataset(genome_fa, elements_tsv, chroms, classes, return_idx_orig=True)
            num_entries = len(dataset)
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

            grp = f.create_group(mode)
            grp.create_dataset("seqs", (num_entries, 500, 4), dtype=np.uint8, shuffle=True, compression="gzip", fletcher32=True, chunks=(1024, 500, 4))
            grp.create_dataset("labels", (num_entries,), dtype=np.uint8, shuffle=True, compression="gzip", fletcher32=True, chunks=(1024,))
            grp.create_dataset("idxs", (num_entries,), dtype=np.uint32, shuffle=True, compression="gzip", fletcher32=True, chunks=(1024,))

            for i, (seq, label, idx) in enumerate(loader):
                start = i * batch_size
                end = start + len(seq)
                grp["seqs"][start:end] = seq.numpy()
                grp["labels"][start:end] = label.numpy()
                grp["idxs"][start:end] = idx.numpy()

    



