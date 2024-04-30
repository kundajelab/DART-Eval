import os
import sys

from torch.utils.data import DataLoader

from ....training import PeaksEmbeddingsDataset, CNNSlicedEmbeddingsPredictor, train_predictor


if __name__ == "__main__":
    resume_checkpoint = int(sys.argv[1]) if len(sys.argv) > 1 else None

    model_name = "hyenadna-large-1m-seqlen-hf"
    peaks_h5 = f"/scratch/groups/akundaje/dnalm_benchmark/embeddings/peak_classification/{model_name}.h5"
    elements_tsv = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/peaks_by_cell_label_unique_dataloader_format.tsv"

    batch_size = 2048
    num_workers = 0
    prefetch_factor = None
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

    input_channels = 256
    hidden_channels = 32
    kernel_size = 8

    crop = 557
    
    lr = 2e-3
    num_epochs = 150

    # out_dir = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/classifiers/ccre_test_regions_500_jitter_50/DNABERT-2-117M/v0"
    out_dir = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/classifiers/peak_classification/{model_name}/v0"
    # out_dir = f"/home/atwang/dnalm_bench_data/predictors/cell_line_2114/{model_name}/{cell_line}/v3"
 
    os.makedirs(out_dir, exist_ok=True)

    classes = {
        "GM12878": 0,
        "H1ESC": 1,
        "HEPG2": 2,
        "IMR90": 3,
        "K562": 4
    } 

    train_dataset = PeaksEmbeddingsDataset(peaks_h5, elements_tsv, chroms_train, classes)
    val_dataset = PeaksEmbeddingsDataset(peaks_h5, elements_tsv, chroms_val, classes)

    model = CNNSlicedEmbeddingsPredictor(input_channels, hidden_channels, kernel_size, out_channels=len(classes))
    train_predictor(train_dataset, val_dataset, model, num_epochs, out_dir, batch_size, lr, num_workers, prefetch_factor, device, progress_bar=True, resume_from=resume_checkpoint)
