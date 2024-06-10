import os
import sys

import torch

from ....finetune import ChromatinEndToEndDataset, evaluate_finetuned_chromatin_model, GENALMLoRAModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

work_dir = os.environ.get("DART_WORK_DIR", "")
cache_dir = os.environ.get("DART_CACHE_DIR")

if __name__ == "__main__":
    cell_line = sys.argv[1] #cell line name
    eval_mode = sys.argv[2] if len(sys.argv) > 2 else "test"

    model_name = "gena-lm-bert-large-t2t"

    genome_fa = os.path.join(work_dir, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")

    peaks_tsv = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_peaks.bed")
    idr_peaks_tsv = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/cell_line_idr_peaks/{cell_line}.bed")
    nonpeaks_tsv = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_nonpeaks.bed")
    assay_bw = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/bigwigs/cell_line_data/{cell_line}_unstranded.bw")

    batch_size = 512
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


    emb_channels = 1024

    crop = 557

    lora_rank = 8
    lora_alpha = 2 * lora_rank
    lora_dropout = 0.05

    model_dir = os.path.join(work_dir, f"task_4_chromatin_activity/supervised_models/fine_tuned/{model_name}/{cell_line}")
    checkpoint_nums = {
        "GM12878": 13,
        "H1ESC": 7,
        "HEPG2": 14,
        "IMR90": 8,
        "K562": 11
    }  
    checkpoint_num = checkpoint_nums[cell_line]    
    checkpoint_path = os.path.join(model_dir, f"checkpoint_{checkpoint_num}.pt") 

    out_dir = os.path.join(work_dir, f"task_4_chromatin_activity/supervised_model_outputs/fine_tuned/{model_name}/{cell_line}")


    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"eval_{eval_mode}.json")

    pos_dataset = ChromatinEndToEndDataset(genome_fa, assay_bw, peaks_tsv, modes[eval_mode], crop, cache_dir=cache_dir)
    idr_dataset = ChromatinEndToEndDataset(genome_fa, assay_bw, idr_peaks_tsv, modes[eval_mode], crop, cache_dir=cache_dir)
    neg_dataset = ChromatinEndToEndDataset(genome_fa, assay_bw, nonpeaks_tsv, modes[eval_mode], crop, cache_dir=cache_dir)

    model = GENALMLoRAModel(model_name, lora_rank, lora_alpha, lora_dropout, 1)
    checkpoint_resume = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_resume, strict=False)
    metrics = evaluate_finetuned_chromatin_model(pos_dataset, idr_dataset, neg_dataset, model, batch_size, out_path,
                                       num_workers, prefetch_factor, device, progress_bar=True)
    
    for k, v in metrics.items():
        print(f"{k}: {v}")