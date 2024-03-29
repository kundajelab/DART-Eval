from ...embeddings import MistralDNAVariantEmbeddingExtractor
from ...components import VariantDataset
import os
import polars as pl
from .variant_tasks import load_embeddings_and_compute_cosine_distance

if __name__ == "__main__":
    model_name = "Mistral-DNA-v0.1"
    # genome_fa = "/mnt/data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    # genome_fa = "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    genome_fa = "/oak/stanford/groups/akundaje/soumyak/refs/hg19/male.hg19.fa"
    # variants_bed = "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/Afr.ASB.CaQTLS.tsv"
    variants_bed = "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/gm12878.dsqtls.benchmarking.tsv"
    batch_size = 2048
    num_workers = 4
    seed = 0
    device = "cuda"
    chroms=None

    h5_file = "mistral.gm12878.dsqtls.benchmarking.variant_embeddings.h5"
    out_dir = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/embeddings/variant_embeddings/Mistral-DNA/"
    # out_dir = "/scratch/groups/akundaje/dnalm_benchmark/embeddings/variant_embeddings/Mistral-DNA/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, h5_file)

    dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
    extractor = MistralDNAVariantEmbeddingExtractor(model_name, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)

    cosine_distances  = load_embeddings_and_compute_cosine_distance(out_dir, h5_file, progress_bar=True)
    print("Number of SNPs =", dataset.__len__())
    print("Number of Cosine Distances = ", len(cosine_distances))

    df = dataset.elements_df
    cos_dist = pl.Series('cosine_distances', cosine_distances)
    df = df.with_columns(cos_dist)
    df.write_csv(os.path.join(out_dir, "mistral.gm12878.dsqtls.benchmarking.cosine_distances.tsv"), separator="\t")
A
