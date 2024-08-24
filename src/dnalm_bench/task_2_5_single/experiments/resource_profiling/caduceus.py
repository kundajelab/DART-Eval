import os
import sys

from ...finetune import ChromatinEndToEndDataset
from ...profile import profile_model_resources, CaduceusModel

work_dir = os.environ.get("DART_WORK_DIR", "")
cache_dir = os.environ.get("DART_CACHE_DIR")

if __name__ == "__main__":
    model_name = "caduceus-ps_seqlen-131k_d_model-256_n_layer-16"

    genome_fa = os.path.join(work_dir, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    
    cell_line = "GM12878"
    peaks_tsv = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_peaks.bed")
    nonpeaks_tsv = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_nonpeaks.bed")
    assay_bw = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/bigwigs/{cell_line}_unstranded.bw")

    batch_size = 16
    num_workers = 4
    prefetch_factor = 2
    seed = 0
    device = "cuda"

    crop = 557

    num_batches_warmup = 100

    out_dir = os.path.join(work_dir, "resource_profiling")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}.json")

    dataset = ChromatinEndToEndDataset(genome_fa, assay_bw, peaks_tsv, None, crop, cache_dir=cache_dir)

    model = CaduceusModel(model_name, 1)

    metrics = profile_model_resources(dataset, model, batch_size, num_batches_warmup, 
                                      out_path, num_workers, prefetch_factor, device, progress_bar=True)
    
    for k, v in metrics.items():
        print(f"{k}: {v}")