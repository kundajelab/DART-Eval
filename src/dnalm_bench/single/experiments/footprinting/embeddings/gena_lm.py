import os
import sys

from ....embeddings import GENALMEmbeddingExtractor
from ....components import FootprintingDataset


if __name__ == "__main__":
    model_name = "gena-lm-bert-large-t2t"
    seq_table = os.path.join(root_output_dir, f"task_2_footprinting/processed_data/footprint_dataset_350_v1.txt")
    batch_size = 64
    num_workers = 0
    seed = 0
    device = "cuda"

    out_dir = os.path.join(root_output_dir, f"task_2_footprinting/outputs/embeddings/footprinting_350/{model_name}/")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"embeddings.h5")

    dataset = FootprintingDataset(seq_table, seed)
    extractor = GENALMEmbeddingExtractor(model_name, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)
