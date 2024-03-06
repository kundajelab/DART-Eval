import argparse
from tqdm.auto import tqdm
from scipy import spatial
from ...embeddings import MistralDNAVariantEmbeddingExtractor
from ...components import VariantDataset
import os

if __name__ == "__main__":
    model_name = "Mistral-DNA-v0.1"
    # genome_fa = "/mnt/data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    genome_fa = "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    variants_bed = "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/Afr.CaQTLS.tsv"
    batch_size = 2048
    num_workers = 4
    seed = 0
    device = "cuda"
    chroms=None

    out_dir = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/embeddings/variant_embeddings/Mistral-DNA/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mistral.Afr.CaQTLs.variant_embeddings.h5")

    dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
    extractor = MistralDNAVariantEmbeddingExtractor(model_name, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)

