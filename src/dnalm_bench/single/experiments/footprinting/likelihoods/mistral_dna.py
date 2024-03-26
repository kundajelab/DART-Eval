import os
import sys

from ....evaluators import MistralEvaluator
from ....components import FootprintingDataset


if __name__ == "__main__":
    model_name = "Mistral-DNA-v0.1"
    #genome_fa = "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    seq_table = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/regions/footprinting/footprint_dataset_350.txt"
    # chroms = ["chr22"]
    batch_size = 64
    num_workers = 4
    seed = 0
    device = "cuda"

    out_dir = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/likelihoods/footprinting_350/{model_name}/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"likelihoods.tsv")


    dataset = FootprintingDataset(seq_table, seed)
    evaluator = MistralEvaluator(model_name, batch_size, num_workers, device)
    evaluator.evaluate(dataset, out_path, progress_bar=True)
