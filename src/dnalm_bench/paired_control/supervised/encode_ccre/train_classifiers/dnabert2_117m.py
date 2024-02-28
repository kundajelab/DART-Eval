import os

from torch.utils.data import DataLoader

from ...training import EmbeddingsDataset, CNNEmbeddingsClassifier, train_classifier


if __name__ == "__main__":
    model_name = "DNABERT-2-117M"
    # embeddings_h5 = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/embeddings/ccre_test_regions_500_jitter_50/DNABERT-2-117M.h5"
    embeddings_h5 = "/srv/scratch/atwang/dnalm_benchmark/embeddings/ccre_test_regions_500_jitter_50/DNABERT-2-117M.h5"
    elements_tsv = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/regions/ccre_test_regions_500_jitter_50.bed"

    batch_size = 2048
    num_workers = 8
    prefetch_factor = 16
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

    num_epochs = 100

    # out_dir = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/classifiers/ccre_test_regions_500_jitter_50/DNABERT-2-117M/v0"
    out_dir = "/mnt/lab_data2/atwang/data/dnalm_benchmark/classifiers/ccre_test_regions_500_jitter_50/DNABERT-2-117M/v0"
    os.makedirs(out_dir, exist_ok=True)

    train_dataset = EmbeddingsDataset(embeddings_h5, elements_tsv, chroms_train)
    val_dataset = EmbeddingsDataset(embeddings_h5, elements_tsv, chroms_val)
    model = CNNEmbeddingsClassifier(input_channels, hidden_channels, kernel_size)
    # train_classifier(train_dataset, val_dataset, model, num_epochs, out_dir, batch_size, num_workers, prefetch_factor, device, progress_bar=True)

    train_classifier(train_dataset, val_dataset, model, num_epochs, out_dir, batch_size, num_workers, prefetch_factor, device, 
                     progress_bar=True, resume_from=os.path.join(out_dir, "checkpoint_28.pt"))