import os
import sys

from ....evaluators import GenaLMEvaluator
from ....components import FootprintingDataset


if __name__ == "__main__":
    model_name = "gena-lm-bert-large-t2t"
    seq_table = os.path.join(root_output_dir, f"task_2_footprinting/processed_data/footprint_dataset_350_v1.txt")
    batch_size = 64
    num_workers = 0
    seed = 0
    device = "cuda"

    out_dir = os.path.join(root_output_dir, f"task_2_footprinting/outputs/likelihoods/footprinting_350/{model_name}/")
    # out_dir = f"/srv/scratch/patelas/{model_name}/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"likelihoods.tsv")


    dataset = FootprintingDataset(seq_table, seed)
    evaluator = GenaLMEvaluator(model_name, batch_size, num_workers, device)
    evaluator.evaluate(dataset, out_path, progress_bar=True)
