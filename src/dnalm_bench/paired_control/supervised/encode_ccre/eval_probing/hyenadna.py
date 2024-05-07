import os
import sys

import torch

from ...training import EmbeddingsDataset, CNNSlicedEmbeddingsClassifier, evaluate_probing_classifier

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    eval_mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    model_name = "hyenadna-large-1m-seqlen-hf"

    embeddings_h5 = f"/scratch/groups/akundaje/dnalm_benchmark/embeddings/ccre_test_regions_350_jitter_0/{model_name}.h5"
    elements_tsv = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/regions/ccre_test_regions_350_jitter_0.bed"

    batch_size = 2048
    num_workers = 4
    prefetch_factor = 2
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

    input_channels = 256
    hidden_channels = 32
    kernel_size = 8

    model_dir = f"/scratch/groups/akundaje/dnalm_benchmark/classifiers/ccre_test_regions_350_jitter_0/{model_name}/v1"
    checkpoint_num = None
    checkpoint_path = os.path.join(model_dir, f"checkpoint_{checkpoint_num}.pt")

    out_dir = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/encode_ccre/eval_probing/ccre_test_regions_350_jitter_0/{model_name}"    

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"eval_{eval_mode}.json")

    test_dataset = EmbeddingsDataset(embeddings_h5, elements_tsv, modes[eval_mode])

    model = CNNSlicedEmbeddingsClassifier(input_channels, hidden_channels, kernel_size)
    checkpoint_resume = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_resume, strict=False)
    metrics = evaluate_probing_classifier(test_dataset, model, out_path, batch_size, num_workers, prefetch_factor, device, progress_bar=True)
    
    for k, v in metrics.items():
        print(f"{k}: {v}")