import os
import sys

from ....embeddings import MistralDNAEmbeddingExtractor
from ....components import SimpleSequence


if __name__ == "__main__":
    model_name = "Mistral-DNA-v0.1"
    #genome_fa = "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    genome_fa = "/mnt/data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    elements_tsv = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/regions/cell_line_expanded_peaks/GM12878_peaks.bed"
    cell_line = "GM12878"
    # chroms = ["chr22"]
    chroms = None
    batch_size = 64
    num_workers = 4
    seed = 0
    device = "cuda"

    out_dir = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/embeddings/cell_line_2114/Mistral-DNA-v0.1/"
    # out_dir = "/mnt/lab_data2/atwang/data/dnalm_benchmark/embeddings/ccre_test_regions_500_jitter_50"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cell_line}.h5")

    dataset = SimpleSequence(genome_fa, elements_tsv, chroms, seed)
    extractor = MistralDNAEmbeddingExtractor(model_name, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)
