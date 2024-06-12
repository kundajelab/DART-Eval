import os
import sys
import pandas as pd
import numpy as np

import torch

from ....training import PeaksEmbeddingsDataset, CNNSequenceBaselinePredictor, eval_peak_classifier, LargeCNNSlicedEmbeddingsPredictor

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    eval_mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    model_name = "probing_head_like"
    peaks_h5 = os.path.join(root_output_dir,f"task_3_peak_classification/embeddings/{model_name}.h5")
    elements_tsv = os.path.join(root_output_dir,"task_3_peak_classification/processed_inputs/peaks_by_cell_label_unique_dataloader_format.tsv")

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

    input_channels = 4
    hidden_channels = 32
    kernel_size = 8
    emb_channels = 256
    init_kernel_size = 41
    pos_channels = 1

    crop = 557
    seq_len = 500

    crop = 557

    out_dir = os.path.join(root_output_dir,f"task_3_peak_classification/supervised_model_outputs/ab_initio/{model_name}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"eval_{eval_mode}.json")

    model_dir = os.path.join(root_output_dir, f"task_3_peak_classification/supervised_models/ab_initio/{model_name}")
    train_log = f"{model_dir}/train.log"
    df = pd.read_csv(train_log, sep="\t")
    checkpoint_num = int(df["epoch"][np.argmin(df["val_loss"])])
    print(checkpoint_num)
    checkpoint_path = os.path.join(model_dir, f"checkpoint_{checkpoint_num}.pt")
    print(checkpoint_path)

    classes = {
        "GM12878": 0,
        "H1ESC": 1,
        "HEPG2": 2,
        "IMR90": 3,
        "K562": 4
    } 

    test_dataset = PeaksEmbeddingsDataset(peaks_h5, elements_tsv, modes[eval_mode], classes)

    model = CNNSequenceBaselinePredictor(emb_channels, hidden_channels, kernel_size, seq_len, init_kernel_size, pos_channels, out_channels=len(classes))
    checkpoint_resume = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_resume)
        
    metrics = eval_peak_classifier(test_dataset, model, out_path, batch_size,
                                    num_workers, prefetch_factor, device, progress_bar=True)
    
    for k, v in metrics.items():
        print(f"{k}: {v}")
