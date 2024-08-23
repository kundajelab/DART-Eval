import os
import sys

from ...finetune import ChromatinEndToEndDataset
from ...profile import profile_model_resources, DNABERT2Model

work_dir = os.environ.get("DART_WORK_DIR", "")
cache_dir = os.environ.get("DART_CACHE_DIR")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    model_name = "DNABERT-2-117M"

    genome_fa = os.path.join(work_dir, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    
    cell_line = "GM12878"
    peaks_tsv = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_peaks.bed")
    nonpeaks_tsv = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_nonpeaks.bed")
    assay_bw = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/bigwigs/{cell_line}_unstranded.bw")

    batch_size = 64
    num_workers = 4
    prefetch_factor = 2
    seed = 0
    device = "cuda"

    crop = 557

    num_batches_warmup = 100

    out_dir = os.path.join(work_dir, f"resource_profiling/{model_name}")

    os.makedirs(out_dir, exist_ok=True)

    dataset = ChromatinEndToEndDataset(genome_fa, assay_bw, peaks_tsv, None, crop, cache_dir=cache_dir)

    model = DNABERT2Model(model_name, 1)

    metrics = profile_model_resources(dataset, model, batch_size, num_batches_warmup, 
                                      out_dir, num_workers, prefetch_factor, device, progress_bar=True)
    
    for k, v in metrics.items():
        print(f"{k}: {v}")