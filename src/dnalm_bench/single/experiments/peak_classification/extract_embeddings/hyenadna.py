import os
import sys

from ....embeddings import HyenaDNAEmbeddingExtractor
from ....components import SimpleSequence


if __name__ == "__main__":
    model_name = "hyenadna-large-1m-seqlen-hf"
    # genome_fa = "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    # genome_fa = "/mnt/data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    genome_fa = "/home/atwang/dnalm_bench_data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"

    elements_tsv = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/peaks_by_cell_label_unique_dataloader_format.tsv"    
    chroms = None
    batch_size = 512
    num_workers = 4
    seed = 0
    device = "cuda"

    out_path = f"/scratch/groups/akundaje/dnalm_benchmark/embeddings/peak_classification/{model_name}.h5"

    cache_dir = "/mnt/disks/ssd-0/dnalm_bench_cache"

    dataset = SimpleSequence(genome_fa, elements_tsv, chroms, seed, cache_dir=cache_dir)
    extractor = HyenaDNAEmbeddingExtractor(model_name, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)
