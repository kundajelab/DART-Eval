import os
import sys

from torch.utils.data import DataLoader

from ....training import AssayEmbeddingsDataset, InterleavedIterableDataset, CNNEmbeddingsPredictor, train_predictor


if __name__ == "__main__":
    resume_checkpoint = int(sys.argv[1]) if len(sys.argv) > 1 else None

    model_name = "DNABERT-2-117M"
    peaks_h5 = f"/scratch/groups/akundaje/dnalm_benchmark/embeddings/peak_classification/{model_name}.h5"
    elements_tsv = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/peaks_by_cell_label_unique_dataloader_format.tsv"

    batch_size = 1024
    num_workers = 4
    prefetch_factor = 2
    # num_workers = 0 ####
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

    input_channels = 768
    hidden_channels = 32
    kernel_size = 8

    crop = 557

    lr = 2e-3
    num_epochs = 150

    out_dir = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/classifiers/peak_classification/{model_name}/v0"
    # out_dir = f"/scratch/groups/akundaje/dnalm_benchmark/predictors/cell_line_2114/{model_name}/{cell_line}/v3"   
    os.makedirs(out_dir, exist_ok=True)

    peaks_train_datset = AssayEmbeddingsDataset(peaks_h5, peaks_tsv, chroms_train, assay_bw, crop=crop)
    nonpeaks_train_dataset = AssayEmbeddingsDataset(nonpeaks_h5, nonpeaks_tsv, chroms_train, assay_bw, crop=crop, downsample_ratio=10)
    train_dataset = InterleavedIterableDataset([peaks_train_datset, nonpeaks_train_dataset])

    peaks_val_dataset = AssayEmbeddingsDataset(peaks_h5, peaks_tsv, chroms_val, assay_bw, crop=crop)
    nonpeaks_val_dataset = AssayEmbeddingsDataset(nonpeaks_h5, nonpeaks_tsv, chroms_val, assay_bw, crop=crop)
    val_dataset = InterleavedIterableDataset([peaks_val_dataset, nonpeaks_val_dataset])

    model = CNNEmbeddingsPredictor(input_channels, hidden_channels, kernel_size)
    train_predictor(train_dataset, val_dataset, model, num_epochs, out_dir, batch_size, lr, num_workers, prefetch_factor, device, progress_bar=True, resume_from=resume_checkpoint)
