import os
import sys

from ....embeddings import NucleotideTransformerEmbeddingExtractor
from ....components import SimpleSequence


if __name__ == "__main__":
    model_name = "nucleotide-transformer-v2-500m-multi-species"
    genome_fa = "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    # genome_fa = "/mnt/data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    cell_line = sys.argv[1] #cell line name
    category = sys.argv[2] #peaks, nonpeaks, or idr
    if category == "idr":
        elements_tsv = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/regions/cell_line_idr_peaks/{cell_line}.bed"
    else:
        elements_tsv = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/regions/cell_line_expanded_peaks/{cell_line}_{category}.bed"
    # chroms = ["chr22"]
    chroms = None
    batch_size = 64
    num_workers = 4
    seed = 0
    device = "cuda"

    out_dir = f"/scratch/groups/akundaje/chrombench/synapse/task_4_chromatin_activity/embeddings/{model_name}/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cell_line}_{category}.h5")

    dataset = SimpleSequence(genome_fa, elements_tsv, chroms, seed)
    extractor = NucleotideTransformerEmbeddingExtractor(model_name, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)
