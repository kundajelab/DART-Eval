import os
import sys

from ....embeddings import HyenaDNAEmbeddingExtractor
from ....components import SimpleSequence


if __name__ == "__main__":
    model_name = "hyenadna-large-1m-seqlen-hf"
    # genome_fa = "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    # genome_fa = "/mnt/data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    genome_fa = "/home/atwang/dnalm_bench_data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"

    cell_line = sys.argv[1] #cell line name
    category = sys.argv[2] #peaks or nonpeaks
    # elements_tsv = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/regions/cell_line_expanded_peaks/{cell_line}_{category}.bed"
    elements_tsv = f"/home/atwang/dnalm_bench_data/cell_line_expanded_peaks/{cell_line}_{category}.bed"
    # chroms = ["chr22"]
    chroms = None
    batch_size = 512
    num_workers = 4
    seed = 0
    device = "cuda"

    # out_dir = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/embeddings/cell_line_2114/{model_name}/"
    # out_dir = f"/scratch/groups/akundaje/dnalm_benchmark/embeddings/cell_line_2114/{model_name}/"
    out_dir = f"/home/atwang/dnalm_bench_data/embeddings/cell_line_2114/{model_name}/"
    # out_dir = "/mnt/lab_data2/atwang/data/dnalm_benchmark/embeddings/ccre_test_regions_500_jitter_50"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cell_line}_{category}.h5")

    cache_dir = "/mnt/disks/ssd-0/dnalm_bench_cache"

    dataset = SimpleSequence(genome_fa, elements_tsv, chroms, seed, cache_dir=cache_dir)
    extractor = HyenaDNAEmbeddingExtractor(model_name, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)
