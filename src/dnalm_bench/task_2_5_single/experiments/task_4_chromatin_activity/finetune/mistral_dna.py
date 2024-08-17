import os
import sys

from ....finetune import ChromatinEndToEndDataset, train_finetuned_chromatin_model, MistralDNALoRAModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

work_dir = os.environ.get("DART_WORK_DIR", "")
cache_dir = os.environ.get("DART_CACHE_DIR")

if __name__ == "__main__":
    cell_line = sys.argv[1] #cell line name
    resume_checkpoint = int(sys.argv[2]) if len(sys.argv) > 2 else None

    model_name = "Mistral-DNA-v1-1.6B-hg38"

    genome_fa = os.path.join(work_dir, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")

    peaks_tsv = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_peaks.bed")
    nonpeaks_tsv = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_nonpeaks.bed")
    assay_bw = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/bigwigs/{cell_line}_unstranded.bw")

    batch_size = 24
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

    emb_channels = 768

    crop = 557

    lora_rank = 8
    lora_alpha = 2 * lora_rank
    lora_dropout = 0.05

    accumulate = 4
    
    lr = 1e-4
    wd = 0.01
    num_epochs = 15

    out_dir = os.path.join(work_dir, f"task_4_chromatin_activity/supervised_models/fine_tuned/{model_name}/{cell_line}")  

    os.makedirs(out_dir, exist_ok=True)

    train_pos_dataset = ChromatinEndToEndDataset(genome_fa, assay_bw, peaks_tsv, chroms_train, crop, cache_dir=cache_dir)
    train_neg_dataset = ChromatinEndToEndDataset(genome_fa, assay_bw, nonpeaks_tsv, chroms_train, crop, cache_dir=cache_dir, downsample_ratio=10)
    val_pos_dataset = ChromatinEndToEndDataset(genome_fa, assay_bw, peaks_tsv, chroms_val, crop, cache_dir=cache_dir)
    val_neg_dataset = ChromatinEndToEndDataset(genome_fa, assay_bw, nonpeaks_tsv, chroms_val, crop, cache_dir=cache_dir)

    model = MistralDNALoRAModel(model_name, lora_rank, lora_alpha, lora_dropout, 1)
    train_finetuned_chromatin_model(train_pos_dataset, train_neg_dataset, val_pos_dataset, val_neg_dataset, model, 
                                    num_epochs, out_dir, batch_size, lr, wd, accumulate, num_workers, prefetch_factor, device, 
                                    progress_bar=True, resume_from=resume_checkpoint)