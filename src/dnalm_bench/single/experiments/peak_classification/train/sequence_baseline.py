import os
import sys

from torch.utils.data import DataLoader

from ....training import PeaksEmbeddingsDataset, CNNSequenceBaselinePredictor, train_predictor


if __name__ == "__main__":
    resume_checkpoint = int(sys.argv[1]) if len(sys.argv) > 1 else None

    model_name = "sequence_baseline"
    peaks_h5 = f"/scratch/groups/akundaje/dnalm_benchmark/embeddings/peak_classification/{model_name}.h5"
    elements_tsv = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/peaks_by_cell_label_unique_dataloader_format.tsv"

    batch_size = 2048
    # batch_size = 1024
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

    n_filters = 64

    emb_channels = 256
    hidden_channels = 32
    pos_channels = 1
    kernel_size = 8
    init_kernel_size = 41

    seq_len = 2114

    # lr = 5e-4
    lr = 1e-3
    # lr = 2e-3

    num_epochs = 150

    # out_dir = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/classifiers/ccre_test_regions_350_jitter_0/{model_name}/v0"
    out_dir = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/classifiers/peak_classification/{model_name}/v0"
    os.makedirs(out_dir, exist_ok=True)

    # cache_dir = f"/srv/scratch/atwang/dnalm_benchmark/cache/embeddings/ccre_test_regions_350_jitter_0/{model_name}"
    cache_dir = None

    classes = {
        "GM12878": 0,
        "H1ESC": 1,
        "HEPG2": 2,
        "IMR90": 3,
        "K562": 4
    } 

    train_dataset = PeaksEmbeddingsDataset(peaks_h5, elements_tsv, chroms_train, classes)
    val_dataset = PeaksEmbeddingsDataset(peaks_h5, elements_tsv, chroms_val, classes)

    model = CNNSequenceBaselinePredictor(emb_channels, hidden_channels, kernel_size, seq_len, init_kernel_size, pos_channels)
    train_predictor(train_dataset, val_dataset, model, num_epochs, out_dir, batch_size, lr, num_workers, prefetch_factor, device, progress_bar=True, resume_from=resume_checkpoint)
