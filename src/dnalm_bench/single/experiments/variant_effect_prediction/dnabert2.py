import argparse
from tqdm.auto import tqdm
from scipy import spatial

from .embedding_variants import load_embeddings_and_compute_cosine_distance
from ...embeddings import DNABERT2VariantEmbeddingExtractor
from ...components import VariantDataset
import os

if __name__ == "__main__":
    model_name = "DNABERT-2-117M"
    # genome_fa = "/mnt/data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    genome_fa = "/scratch/groups/akundaje/dnalm_benchmark/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    variants_bed = "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/Afr.CaQTLS.tsv"
    batch_size = 2048
    num_workers = 4
    seed = 0
    device = "cuda"
    chroms=None

    out_dir = "/scratch/groups/akundaje/dnalm_benchmark/embeddings/variant_embeddings/DNABERT2/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dnabert-117M.Afr.CaQTLs.variant_embeddings.h5")

    dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
    extractor = DNABERT2VariantEmbeddingExtractor(model_name, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)

    cosine_distances  = load_embeddings_and_compute_cosine_distance(out_dir)

    df = dataset.elements_df
    df["cosine_distances"]=cosine_distances # make sure that the cosine distances correspond to the correct allele
    df.write_csv(os.path.join(out_dir, "dnabert2.Afr.CaQTLs.cosine_distances.tsv"))

