import os

from ...embeddings import HyenaDNAEmbeddingExtractor
from ....components import PairedControlDataset


if __name__ == "__main__":
    model_name = "hyenadna-large-1m-seqlen-hf"
    genome_fa = "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    # genome_fa = "/mnt/data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    elements_tsv = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/regions/ccre_test_regions_350_jitter_0.bed"
    # chroms = ["chr22"]
    chroms = None
    batch_size = 1024
    num_workers = 4
    seed = 0
    device = "cuda"

    # out_dir = "/srv/scratch/atwang/dnalm_benchmark/embeddings/ccre_test_regions_350_jitter_0"
    # out_dir = "/mnt/lab_data2/atwang/data/dnalm_benchmark/embeddings/ccre_test_regions_350_jitter_0"
    out_dir = "/scratch/groups/akundaje/dnalm_benchmark/embeddings/ccre_test_regions_350_jitter_0"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}.h5")

    dataset = PairedControlDataset(genome_fa, elements_tsv, chroms, seed)
    extractor = HyenaDNAEmbeddingExtractor(model_name, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)
