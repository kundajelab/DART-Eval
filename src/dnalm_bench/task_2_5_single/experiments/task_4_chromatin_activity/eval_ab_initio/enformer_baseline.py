import os
import sys

import numpy as np
import pandas as pd

from ....enformer_utils import *
root_output_dir = os.environ.get("DART_WORK_DIR", "")

cell_line = sys.argv[1]
genome_fa = os.path.join(root_output_dir, f"refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")

bigwig_file = os.path.join(root_output_dir, f"task_4_chromatin_activity/processed_data/bigwigs/{cell_line}_unstranded.bw")
batch_size = 1

head_dict = {"GM12878": 69, "HEPG2": 234, "IMR90": 405, "K562": 121, "H1ESC":70}
head = head_dict[cell_line]
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



peaks_tsv = os.path.join(root_output_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_peaks.bed")
idr_peaks_tsv = os.path.join(root_output_dir, f"task_4_chromatin_activity/processed_data/cell_line_idr_peaks/{cell_line}.bed")
nonpeaks_tsv = os.path.join(root_output_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_nonpeaks.bed")
out_file = os.path.join(root_output_dir, f"task_4_chromatin_activity/supervised_model_outputs/enformer/{cell_line}_metrics.json")

#chroms_test = chroms_train + chroms_val + chroms_test
# peaks_tsv = "/users/patelas/scratch/GM12878_peaks_enformertest_dataset.bed"
# nonpeaks_tsv = "/users/patelas/scratch/GM12878_nonpeaks_enformertest_dataset.bed"
# idr_peaks_tsv = "/users/patelas/scratch/GM12878_idr_enformertest_dataset.bed"
# out_file = "/users/patelas/scratch/GM12878_combinedtest_metrics.json"


print("Loading Datasets")
print("Loading peak dataset")
pos_peak_dataset = EnformerPeakDataset(peaks_tsv, bigwig_file, genome_fa, chroms_test, batch_size)
print("Loading negative dataset")
neg_peak_dataset = EnformerPeakDataset(nonpeaks_tsv, bigwig_file, genome_fa, chroms_test, batch_size)
print("Loading IDR dataset")
idr_peak_dataset = EnformerPeakDataset(idr_peaks_tsv, bigwig_file, genome_fa, chroms_test, batch_size)

print(len(pos_peak_dataset), len(neg_peak_dataset), len(idr_peak_dataset))

print("Loading Model")
model = Enformer('https://tfhub.dev/deepmind/enformer/1')

print("Predicting on Peaks")
true_pos, pred_pos = get_counts_predictions(model, pos_peak_dataset, head)
print("Predicting on Negatives")
true_neg, pred_neg = get_counts_predictions(model, neg_peak_dataset, head)
print("Predicting on IDR Peaks")
true_idr, pred_idr = get_counts_predictions(model, idr_peak_dataset, head)

print("Calculating Metrics")
metrics = calc_metrics(true_pos, pred_pos, true_neg, pred_neg, true_idr, pred_idr, out_file)


