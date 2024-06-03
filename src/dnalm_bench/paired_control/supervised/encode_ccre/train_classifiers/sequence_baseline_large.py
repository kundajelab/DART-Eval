import os
import sys

from torch.utils.data import DataLoader

from ...training import EmbeddingsDataset, train_classifier, LargeCNNSequenceBaselineClassifier


if __name__ == "__main__":
    resume_checkpoint = int(sys.argv[1]) if len(sys.argv) > 1 else None

    model_name = "sequence_baseline_large"
    # embeddings_h5 = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/embeddings/ccre_test_regions_350_jitter_0/{model_name}.h5"
    # embeddings_h5 = f"/scratch/groups/akundaje/dnalm_benchmark/embeddings/ccre_test_regions_350_jitter_0/{model_name}.h5"
    # elements_tsv = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/regions/ccre_test_regions_350_jitter_0.bed"

    embeddings_h5 = f"/scratch/groups/akundaje/chrombench/synapse/task_1_ccre/embeddings/sequence_baseline.h5"
    elements_tsv = f"/scratch/groups/akundaje/chrombench/synapse/task_1_ccre/processed_data/ENCFF420VPZ_processed.tsv"

    batch_size = 512
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

    n_filters_trunk = 512
    n_residual_trunk = 7
    # n_layers = 7
    # n_layers_dil = 7

    # input_channels = 256
    emb_channels = 256
    hidden_channels = 32
    pos_channels = 1
    kernel_size = 8
    init_kernel_size = 41


    seq_len = 350

    # n_layers_trunk = 7

    # lr = 5e-4
    lr = 1e-3
    # lr = 2e-3

    num_epochs = 150

    # out_dir = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/classifiers/ccre_test_regions_350_jitter_0/{model_name}/v0"
    # out_dir = f"/scratch/groups/akundaje/dnalm_benchmark/classifiers/ccre_test_regions_350_jitter_0/{model_name}/v30"
    out_dir = f"/scratch/groups/akundaje/chrombench/synapse/task_1_ccre/supervised_classifiers/{model_name}/v1" 
    os.makedirs(out_dir, exist_ok=True)

    # cache_dir = f"/srv/scratch/atwang/dnalm_benchmark/cache/embeddings/ccre_test_regions_350_jitter_0/{model_name}"
    cache_dir = None

    train_dataset = EmbeddingsDataset(embeddings_h5, elements_tsv, chroms_train, cache_dir=cache_dir)
    val_dataset = EmbeddingsDataset(embeddings_h5, elements_tsv, chroms_val, cache_dir=cache_dir)
    model = LargeCNNSequenceBaselineClassifier(emb_channels, hidden_channels, kernel_size, seq_len, init_kernel_size, pos_channels, n_filters_trunk, n_residual_trunk)
    # model = CNNSequenceBaselineClassifier(n_filters, n_layers)
    # model = CNNSequenceBaselineClassifier(input_channels, hidden_channels, kernel_size, n_layers_trunk)

    # print(f"Parameter count: {sum(p.numel() for p in model.parameters())}")
    train_classifier(train_dataset, val_dataset, model, num_epochs, out_dir, batch_size, lr, num_workers, prefetch_factor, device, progress_bar=True, resume_from=resume_checkpoint)