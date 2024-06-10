import os
import sys

from ....finetune import train_finetuned_classifier, LargeCNNClassifier
from ....components import PairedControlDataset

work_dir = os.environ.get("DART_WORK_DIR", "")
cache_dir = os.environ.get("DART_CACHE_DIR")

if __name__ == "__main__":
    resume_checkpoint = int(sys.argv[1]) if len(sys.argv) > 1 else None

    model_name = "chrombpnet_like"

    genome_fa = os.path.join(work_dir, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    elements_tsv = os.path.join(work_dir, f"task_1_ccre/processed_inputs/ENCFF420VPZ_processed.tsv")

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

    accumulate = 1
    
    lr = 1e-4
    wd = 0
    num_epochs = 150

    n_filters = 512
    n_residual_convs = 7
    output_channels = 2
    seq_len = 330

    out_dir = os.path.join(work_dir, f"task_1_ccre/supervised_models/ab_initio/{model_name}")  

    os.makedirs(out_dir, exist_ok=True)

    train_dataset = PairedControlDataset(genome_fa, elements_tsv, chroms_train, seed, cache_dir=cache_dir)
    val_dataset = PairedControlDataset(genome_fa, elements_tsv, chroms_val, seed, cache_dir=cache_dir)

    model = LargeCNNClassifier(4, n_filters, n_residual_convs, output_channels, seq_len)
    train_finetuned_classifier(train_dataset, val_dataset, model, num_epochs, out_dir, 
                               batch_size, lr, wd, accumulate, num_workers, prefetch_factor, 
                               device, progress_bar=True, resume_from=resume_checkpoint)