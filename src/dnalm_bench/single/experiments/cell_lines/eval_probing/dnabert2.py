import os
import sys

import torch
import numpy as np
import pandas as pd

from ....training import AssayEmbeddingsDataset, evaluate_chromatin_model, CNNEmbeddingsPredictor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    cell_line = sys.argv[1] #cell line name
    eval_mode = sys.argv[2] if len(sys.argv) > 2 else "test"

    model_name = "DNABERT-2-117M"

    peaks_h5 = f"/scratch/groups/akundaje/dnalm_benchmark/embeddings/cell_line_2114/{model_name}/{cell_line}_peaks.h5"
    idr_h5 = f"/scratch/groups/akundaje/dnalm_benchmark/embeddings/cell_line_2114/{model_name}/{cell_line}_idr.h5"
    nonpeaks_h5 = f"/scratch/groups/akundaje/dnalm_benchmark/embeddings/cell_line_2114/{model_name}/{cell_line}_nonpeaks.h5"
    peaks_tsv = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/regions/cell_line_expanded_peaks/{cell_line}_peaks.bed"
    idr_peaks_tsv = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/regions/cell_line_idr_peaks/{cell_line}.bed"
    nonpeaks_tsv = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/regions/cell_line_expanded_peaks/{cell_line}_nonpeaks.bed"
    assay_bw = f"/scratch/groups/akundaje/dnalm_benchmark/cell_line_data/{cell_line}_unstranded.bw"

    batch_size = 1024
    num_workers = 0
    prefetch_factor = None
    seed = 0
    device = "cuda"

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

    modes = {"train": chroms_train, "val": chroms_val, "test": chroms_test}

    input_channels = 768
    hidden_channels = 32
    kernel_size = 8

    crop = 557

    model_dir = f"/scratch/groups/akundaje/chrombench/synapse/task_4_chromatin_activity/supervised_classifiers/probed/{model_name}/{cell_line}/v1/"

    train_log = f"{model_dir}/train.log"
    df = pd.read_csv(train_log, sep="\t")
    checkpoint_num = int(df["epoch"][np.argmin(df["val_loss"])])
    checkpoint_path = os.path.join(model_dir, f"checkpoint_{checkpoint_num}.pt")

    out_dir = f"/scratch/groups/akundaje/chrombench/synapse/task_4_chromatin_activity/outputs/{model_name}/{cell_line}" 
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"eval_{eval_mode}.json")

    pos_dataset = AssayEmbeddingsDataset(peaks_h5, peaks_tsv, modes[eval_mode], assay_bw, crop=crop)
    idr_dataset = AssayEmbeddingsDataset(idr_h5, idr_peaks_tsv, modes[eval_mode], assay_bw, crop=crop)
    neg_dataset = AssayEmbeddingsDataset(nonpeaks_h5, nonpeaks_tsv, modes[eval_mode], assay_bw, crop=crop)

    model = CNNEmbeddingsPredictor(input_channels, hidden_channels, kernel_size)
    checkpoint_resume = torch.load(checkpoint_path)
    # print(checkpoint_resume.keys()) ####
    model.load_state_dict(checkpoint_resume, strict=False)

    metrics = evaluate_chromatin_model(pos_dataset, idr_dataset, neg_dataset, model, batch_size, out_path,
                                       num_workers, prefetch_factor, device, progress_bar=True)
    
    for k, v in metrics.items():
        print(f"{k}: {v}")