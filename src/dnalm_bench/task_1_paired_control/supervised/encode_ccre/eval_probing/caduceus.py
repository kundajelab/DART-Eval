import os
import sys

import torch
import pandas as pd
import numpy as np

from ...training import EmbeddingsDataset, CNNEmbeddingsClassifier, evaluate_probing_classifier

os.environ["TOKENIZERS_PARALLELISM"] = "false"
work_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    eval_mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    model_name = "caduceus-ps_seqlen-131k_d_model-256_n_layer-16"

    embeddings_h5 = os.path.join(work_dir, f"task_1_ccre/embeddings/{model_name}.h5")
    elements_tsv = os.path.join(work_dir, "task_1_ccre/processed_inputs/ENCFF420VPZ_processed.tsv")

    batch_size = 2048
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

    input_channels = 1024
    hidden_channels = 32
    kernel_size = 8

    model_dir = os.path.join(work_dir, f"task_1_ccre/supervised_models/probed/{model_name}")

    train_log = f"{model_dir}/train.log"
    df = pd.read_csv(train_log, sep="\t")
    checkpoint_num = int(df["epoch"][np.argmin(df["val_loss"])])

    checkpoint_path = os.path.join(model_dir, f"checkpoint_{checkpoint_num}.pt")

    out_dir = os.path.join(work_dir, f"task_1_ccre/supervised_model_outputs/probed/{model_name}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"eval_{eval_mode}.json")

    test_dataset = EmbeddingsDataset(embeddings_h5, elements_tsv, modes[eval_mode])

    model = CNNEmbeddingsClassifier(input_channels, hidden_channels, kernel_size)
    checkpoint_resume = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_resume, strict=False)
    metrics = evaluate_probing_classifier(test_dataset, model, out_path, batch_size, num_workers, prefetch_factor, device, progress_bar=True)
    
    for k, v in metrics.items():
        print(f"{k}: {v}")