import os
import sys
import numpy as np
import polars as pl

from ....evaluators import DNABERT2VariantEmbeddingEvaluator
from ....components import VariantDataset


if __name__ == "__main__":
    dataset = sys.argv[1]

    model_name = "DNABERT-2-117M"
    batch_size = 512
    num_workers = 4
    seed = 0
    device = "cuda"
    chroms=None

    genomes = {
        "gm12878.dsqtls.benchmarking": "/home/atwang/dnalm_bench_data/male.hg19.fa", 
        "Eu.CaQTLS": "/home/atwang/dnalm_bench_data/male.hg19.fa",
        "Afr.ASB.CaQTLS": "/home/atwang/dnalm_bench_data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta",
        "Afr.CaQTLS": "/home/atwang/dnalm_bench_data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    }
    genome_fa = genomes[dataset]

    variants_bed = f"/home/atwang/dnalm_bench_data/variant-benchmarking/{dataset}.tsv" 

    out_dir = f"/home/atwang/dnalm_bench_data/embeddings/variants/{model_name}/"
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, f"{dataset}.tsv")

    allele1_embeddings_path = os.path.join(out_dir, f"{dataset}_allele1_embeddings.npy")
    allele2_embeddings_path = os.path.join(out_dir, f"{dataset}_allele2_embeddings.npy")

    dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
    evaluator = DNABERT2VariantEmbeddingEvaluator(model_name, batch_size, num_workers, device)
    score_df, allele1_embeddings, allele2_embeddings = evaluator.evaluate(dataset, out_path, progress_bar=True)

    df = dataset.elements_df
    scored_df = pl.concat([df, score_df], how="horizontal")
    print(out_path)
    scored_df.write_csv(out_path, separator="\t")

    # Save embeddings
    np.save(allele1_embeddings_path, allele1_embeddings)
    np.save(allele2_embeddings_path, allele2_embeddings)
