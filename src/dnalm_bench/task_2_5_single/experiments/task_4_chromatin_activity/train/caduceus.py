import os
import sys

from torch.utils.data import DataLoader

from ....training import AssayEmbeddingsDataset, InterleavedIterableDataset, CNNEmbeddingsPredictor, train_predictor

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    cell_line = sys.argv[1] #cell line name
    resume_checkpoint = int(sys.argv[2]) if len(sys.argv) > 2 else None

    model_name = "caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    peaks_h5 = os.path.join(root_output_dir, f"task_4_chromatin_activity/embeddings/{model_name}/{cell_line}_peaks.h5")
    nonpeaks_h5 = os.path.join(root_output_dir, f"task_4_chromatin_activity/embeddings/{model_name}/{cell_line}_nonpeaks.h5")
    peaks_tsv = os.path.join(root_output_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_peaks.bed")
    nonpeaks_tsv = os.path.join(root_output_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_nonpeaks.bed")
    assay_bw = os.path.join(root_output_dir, f"task_4_chromatin_activity/processed_data/bigwigs/{cell_line}_unstranded.bw")

    batch_size = 512
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

    input_channels = 512
    hidden_channels = 32
    kernel_size = 3

    crop = 557
    
    lr = 2e-3
    num_epochs = 150

    out_dir = os.path.join(root_output_dir, f"task_4_chromatin_activity/supervised_models/probed/{model_name}/{cell_line}/v1")
 
    os.makedirs(out_dir, exist_ok=True)

    peaks_train_datset = AssayEmbeddingsDataset(peaks_h5, peaks_tsv, chroms_train, assay_bw, crop=crop)
    nonpeaks_train_dataset = AssayEmbeddingsDataset(nonpeaks_h5, nonpeaks_tsv, chroms_train, assay_bw, crop=crop, downsample_ratio=10)
    train_dataset = InterleavedIterableDataset([peaks_train_datset, nonpeaks_train_dataset])

    peaks_val_dataset = AssayEmbeddingsDataset(peaks_h5, peaks_tsv, chroms_val, assay_bw, crop=crop)
    nonpeaks_val_dataset = AssayEmbeddingsDataset(nonpeaks_h5, nonpeaks_tsv, chroms_val, assay_bw, crop=crop)
    val_dataset = InterleavedIterableDataset([peaks_val_dataset, nonpeaks_val_dataset])

    model = CNNEmbeddingsPredictor(input_channels, hidden_channels, kernel_size)
    train_predictor(train_dataset, val_dataset, model, num_epochs, out_dir, batch_size, lr, num_workers, prefetch_factor, device, progress_bar=True, resume_from=resume_checkpoint)
