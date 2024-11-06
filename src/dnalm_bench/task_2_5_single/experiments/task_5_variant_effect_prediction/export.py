import os

import numpy as np
import h5py
from torch.utils.data import DataLoader

from ...finetune import ChromatinEndToEndDataset
from ...components import VariantDataset
import pandas as pd

work_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":

    batch_size = 256
    num_workers = 0
    seed = 0
    device = "cuda"
    chroms=None

    out_path = os.path.join(work_dir, f"task_5_variant_effect_prediction/data.h5")

    with h5py.File(out_path, "w") as f:
        variants_bed = os.path.join(work_dir, "task_5_variant_effect_prediction/input_data/Afr.CaQTLS.tsv")
        genome_fa = os.path.join(work_dir, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
        variants_bed_df = pd.read_csv(variants_bed, sep="\t")
        variants_bed_grp = f.create_group(os.path.basename(variants_bed))
        dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        num_entries = len(dataset)
        
        variants_bed_grp.create_dataset("allele_1_seqs", (num_entries, 2114, 4), dtype=np.uint8, shuffle=False, compression="gzip", fletcher32=True, chunks=(256, 2114, 4))
        variants_bed_grp.create_dataset("allele_2_seqs", (num_entries, 2114, 4), dtype=np.uint8, shuffle=False, compression="gzip", fletcher32=True, chunks=(256, 2114, 4))
        variants_bed_grp.create_dataset("is_causal", (num_entries,), dtype=np.uint8, shuffle=False, compression="gzip", fletcher32=True, chunks=(256,))
        variants_bed_grp.create_dataset("effect_size", (num_entries,), dtype=np.float32, shuffle=False, compression="gzip", fletcher32=True, chunks=(256,))

        for (i, (allele_1_seq, allele_2_seq)) in enumerate(dataloader):
            start = i * batch_size
            end = start + len(allele_1_seq)

            variants_bed_grp["allele_1_seqs"][start:end] = allele_1_seq.numpy()
            variants_bed_grp["allele_2_seqs"][start:end] = allele_2_seq.numpy()
            variants_bed_grp["is_causal"][start:end] = pd.Series(dataset.elements_df["IsUsed"][start:end]).astype(int).to_numpy()
            variants_bed_grp["effect_size"][start:end] = pd.Series(dataset.elements_df["beta"][start:end]).astype("float32").to_numpy()

        variants_bed = os.path.join(work_dir, "task_5_variant_effect_prediction/input_data/yoruban.dsqtls.benchmarking.tsv")
        genome_fa = os.path.join(work_dir, "refs/male.hg19.fa")
        variants_bed_df = pd.read_csv(variants_bed, sep="\t")
        variants_bed_grp = f.create_group(os.path.basename(variants_bed))
        dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        num_entries = len(dataset)
        
        variants_bed_grp.create_dataset("allele_1_seqs", (num_entries, 2114, 4), dtype=np.uint8, shuffle=False, compression="gzip", fletcher32=True, chunks=(256, 2114, 4))
        variants_bed_grp.create_dataset("allele_2_seqs", (num_entries, 2114, 4), dtype=np.uint8, shuffle=False, compression="gzip", fletcher32=True, chunks=(256, 2114, 4))
        variants_bed_grp.create_dataset("is_causal", (num_entries,), dtype=np.uint8, shuffle=False, compression="gzip", fletcher32=True, chunks=(256,))
        variants_bed_grp.create_dataset("effect_size", (num_entries,), dtype=np.float32, shuffle=False, compression="gzip", fletcher32=True, chunks=(256,))

        for (i, (allele_1_seq, allele_2_seq)) in enumerate(dataloader):
            start = i * batch_size
            end = start + len(allele_1_seq)

            variants_bed_grp["allele_1_seqs"][start:end] = allele_1_seq.numpy()
            variants_bed_grp["allele_2_seqs"][start:end] = allele_2_seq.numpy()
            variants_bed_grp["is_causal"][start:end] = pd.Series(dataset.elements_df["var.isused"][start:end]).astype(int).to_numpy()
            variants_bed_grp["effect_size"][start:end] = pd.Series(dataset.elements_df["obs.estimate"][start:end]).astype("float32").to_numpy()

        # for cell_line in cell_lines:
        #     cell_grp = f.create_group(cell_line)
        #     for mode, chroms in zip(["train", "val", "test"], [chroms_train, chroms_val, chroms_test]):
        #         mode_grp = cell_grp.create_group(mode)

        #         peaks_tsv = os.path.join(work_dir, f"task_5_variant_effect_prediction/processed_data/cell_line_expanded_peaks/{cell_line}_peaks.bed")
        #         idr_peaks_tsv = os.path.join(work_dir, f"task_5_variant_effect_prediction/processed_data/cell_line_idr_peaks/{cell_line}.bed")
        #         nonpeaks_tsv = os.path.join(work_dir, f"task_5_variant_effect_prediction/processed_data/cell_line_expanded_peaks/{cell_line}_nonpeaks.bed")
        #         assay_bw = os.path.join(work_dir, f"task_5_variant_effect_prediction/processed_data/bigwigs/{cell_line}_unstranded.bw")

        #         pos_dataset = ChromatinEndToEndDataset(genome_fa, assay_bw, peaks_tsv, chroms, crop, return_idx_orig=True)
        #         idr_dataset = ChromatinEndToEndDataset(genome_fa, assay_bw, idr_peaks_tsv, chroms, crop, return_idx_orig=True)
        #         neg_dataset = ChromatinEndToEndDataset(genome_fa, assay_bw, nonpeaks_tsv, chroms, crop, return_idx_orig=True)

        #         for dataset, peak_set in zip([pos_dataset, idr_dataset, neg_dataset], ["peaks", "idr_peaks", "nonpeaks"]):
        #             num_entries = len(dataset)
        #             loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

        #             grp = mode_grp.create_group(peak_set)
        #             grp.create_dataset("seqs", (num_entries, 2114, 4), dtype=np.uint8, shuffle=True, compression="gzip", fletcher32=True, chunks=(1024, 2114, 4))
        #             grp.create_dataset("counts", (num_entries,), dtype=np.float32, shuffle=True, compression="gzip", fletcher32=True, chunks=(1024,))
        #             grp.create_dataset("idxs", (num_entries,), dtype=np.uint32, shuffle=True, compression="gzip", fletcher32=True, chunks=(1024,))

        #             for i, (seq, signal, idx_orig) in enumerate(loader):
        #                 log1p_counts = np.log1p(signal.numpy().sum(axis=1))

        #                 start = i * batch_size
        #                 end = start + len(seq)

        #                 grp["seqs"][start:end] = seq.numpy()
        #                 grp["counts"][start:end] = log1p_counts
        #                 grp["idxs"][start:end] = idx_orig.numpy()