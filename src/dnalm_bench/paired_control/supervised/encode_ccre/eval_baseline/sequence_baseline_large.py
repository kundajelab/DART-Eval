import os
import sys

import torch

from ....finetune import evaluate_finetuned_classifier, LargeCNNClassifier
from ....components import PairedControlDataset

if __name__ == "__main__":
    eval_mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    model_name = "sequence_baseline_large"

    genome_fa = "/home/atwang/dnalm_bench_data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    elements_tsv = f"/home/atwang/dnalm_bench_data/ccre_test_regions_350_jitter_0.bed"

    batch_size = 4096
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

    # emb_channels = 256

    # crop = 557

    # lora_rank = 8
    # lora_alpha = 2 * lora_rank
    # lora_dropout = 0.05

    n_filters = 512
    n_residual_convs = 7
    output_channels = 2
    seq_len = 330

    # cache_dir = os.environ["L_SCRATCH_JOB"]
    cache_dir = "/mnt/disks/ssd-0/dnalm_bench_cache"

    model_dir = f"/home/atwang/dnalm_bench_data/encode_ccre/classifiers/ccre_test_regions_350_jitter_0/{model_name}/v6"      
    checkpoint_num = 59
    checkpoint_path = os.path.join(model_dir, f"checkpoint_{checkpoint_num}.pt")

    out_dir = f"/home/atwang/dnalm_bench_data/encode_ccre/eval/ccre_test_regions_350_jitter_0/{model_name}"    

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"eval_{eval_mode}.json")

    test_dataset = PairedControlDataset(genome_fa, elements_tsv, modes[eval_mode], seed)

    model = LargeCNNClassifier(4, n_filters, n_residual_convs, output_channels, seq_len)
    checkpoint_resume = torch.load(checkpoint_path)
    # print(checkpoint_resume.keys()) ####
    model.load_state_dict(checkpoint_resume, strict=False)
    metrics = evaluate_finetuned_classifier(test_dataset, model, out_path, batch_size, num_workers, prefetch_factor, device, progress_bar=True)
    
    for k, v in metrics.items():
        print(f"{k}: {v}")