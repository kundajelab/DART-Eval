import os

from ...embeddings import GenaLMEmbeddingExtractor
from ....components import PairedControlDataset


if __name__ == "__main__":
    model_name = "gena-lm-bert-large-t2t"
    genome_fa = "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    # elements_tsv = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/regions/ccre_test_regions_350_jitter_0.bed"
    elements_tsv = "/scratch/groups/akundaje/chrombench/synapse/task_1_ccre/processed_inputs/ENCFF420VPZ_processed.tsv"
    # chroms = ["chr22"]
    chroms = None
    batch_size = 512
    num_workers = 4
    seed = 0
    device = "cuda"

    # out_dir = "/scratch/groups/akundaje/dnalm_benchmark/embeddings/ccre_test_regions_350_jitter_0"
    out_dir = "/scratch/groups/akundaje/chrombench/synapse/task_1_ccre/embeddings/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}.h5")

    dataset = PairedControlDataset(genome_fa, elements_tsv, chroms, seed)
    extractor = GenaLMEmbeddingExtractor(model_name, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)