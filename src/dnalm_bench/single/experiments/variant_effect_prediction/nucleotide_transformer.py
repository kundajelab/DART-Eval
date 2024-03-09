from .variants_tasks import load_embeddings_and_compute_cosine_distance
from ...embeddings import NucleotideTransformerVariantEmbeddingExtractor
from ...components import VariantDataset
import os
import polars as pl

if __name__ == "__main__":
    model_name = "nucleotide-transformer-v2-500m-multi-species"
    # genome_fa = "/mnt/data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    genome_fa = "/scratch/groups/akundaje/dnalm_benchmark/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    variants_bed = "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/Afr.CaQTLS.tsv"
    batch_size = 2048
    num_workers = 4
    seed = 0
    device = "cuda"
    chroms=None

    out_dir = "/scratch/groups/akundaje/dnalm_benchmark/embeddings/variant_embeddings/Nucleotide-Transformer/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "nt-500m.Afr.CaQTLs.variant_embeddings.h5")

    dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
    extractor = NucleotideTransformerVariantEmbeddingExtractor(model_name, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)

    cosine_distances  = load_embeddings_and_compute_cosine_distance(out_dir, progress_bar=True)

    df = dataset.elements_df
    cos_dist = pl.Series('cosine_distances', cosine_distances)
    df = df.with_columns(cos_dist)
    df.write_csv(os.path.join(out_dir, "nt.Afr.CaQTLs.cosine_distances.tsv"), separator="\t")

