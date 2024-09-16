import os

import numpy as np
import h5py
from torch.utils.data import DataLoader

from ...finetune import ChromatinEndToEndDataset

work_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    genome_fa = os.path.join(work_dir, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")

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

    crop = 557
    batch_size = 2048
    cell_lines = ["GM12878", "H1ESC", "HEPG2", "IMR90", "K562"]

    out_path = os.path.join(work_dir, f"task_4_chromatin_activity/data.h5")

    with h5py.File(out_path, "w") as f:
        for cell_line in cell_lines:
            cell_grp = f.create_group(cell_line)
            for mode, chroms in zip(["train", "val", "test"], [chroms_train, chroms_val, chroms_test]):
                mode_grp = cell_grp.create_group(mode)

                peaks_tsv = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_peaks.bed")
                idr_peaks_tsv = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/cell_line_idr_peaks/{cell_line}.bed")
                nonpeaks_tsv = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_nonpeaks.bed")
                assay_bw = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/bigwigs/{cell_line}_unstranded.bw")

                pos_dataset = ChromatinEndToEndDataset(genome_fa, assay_bw, peaks_tsv, chroms, crop, return_idx_orig=True)
                idr_dataset = ChromatinEndToEndDataset(genome_fa, assay_bw, idr_peaks_tsv, chroms, crop, return_idx_orig=True)
                neg_dataset = ChromatinEndToEndDataset(genome_fa, assay_bw, nonpeaks_tsv, chroms, crop, return_idx_orig=True)

                for dataset, peak_set in zip([pos_dataset, idr_dataset, neg_dataset], ["peaks", "idr_peaks", "nonpeaks"]):
                    num_entries = len(dataset)
                    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

                    grp = mode_grp.create_group(peak_set)
                    grp.create_dataset("seqs", (num_entries, 2114, 4), dtype=np.uint8, shuffle=True, compression="gzip", fletcher32=True, chunks=(1024, 2114, 4))
                    grp.create_dataset("counts", (num_entries,), dtype=np.float32, shuffle=True, compression="gzip", fletcher32=True, chunks=(1024,))
                    grp.create_dataset("idxs", (num_entries,), dtype=np.uint32, shuffle=True, compression="gzip", fletcher32=True, chunks=(1024,))

                    for i, (seq, signal, idx_orig) in enumerate(loader):
                        log1p_counts = np.log1p(signal.numpy().sum(axis=1))

                        start = i * batch_size
                        end = start + len(seq)

                        grp["seqs"][start:end] = seq.numpy()
                        grp["counts"][start:end] = log1p_counts
                        grp["idxs"][start:end] = idx_orig.numpy()

                        



