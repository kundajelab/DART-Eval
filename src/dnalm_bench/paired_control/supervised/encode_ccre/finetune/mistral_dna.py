import os
import sys

import torch

# from ....training import AssayEmbeddingsDataset, InterleavedIterableDataset, CNNEmbeddingsPredictor, train_predictor
from ....finetune import train_finetuned_classifier, MistralDNALoRAModel
from ....components import PairedControlDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    resume_checkpoint = int(sys.argv[1]) if len(sys.argv) > 1 else None

    model_name = "Mistral-DNA-v0.1"
    # genome_fa = "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    # genome_fa = "/mnt/data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    genome_fa = "/home/atwang/dnalm_bench_data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    elements_tsv = f"/home/atwang/dnalm_bench_data/ccre_test_regions_350_jitter_0.bed"

    batch_size = 24
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

    emb_channels = 256

    crop = 557

    lora_rank = 8
    lora_alpha = 2 * lora_rank
    lora_dropout = 0.05

    accumulate = 4
    
    lr = 1e-4
    wd = 0.01
    num_epochs = 10

    # cache_dir = os.environ["L_SCRATCH_JOB"]
    cache_dir = "/mnt/disks/ssd-0/dnalm_bench_cache"

    out_dir = f"/home/atwang/dnalm_bench_data/encode_ccre/classifiers_ft/ccre_test_regions_350_jitter_0/{model_name}/v0"      

    os.makedirs(out_dir, exist_ok=True)

    train_dataset = PairedControlDataset(genome_fa, elements_tsv, chroms_train, seed)
    val_dataset = PairedControlDataset(genome_fa, elements_tsv, chroms_val, seed)

    model = MistralDNALoRAModel(model_name, lora_rank, lora_alpha, lora_dropout, 2)
    train_finetuned_classifier(train_dataset, val_dataset, model, num_epochs, out_dir, 
                               batch_size, lr, wd, accumulate, num_workers, prefetch_factor, 
                               device, progress_bar=True, resume_from=resume_checkpoint)