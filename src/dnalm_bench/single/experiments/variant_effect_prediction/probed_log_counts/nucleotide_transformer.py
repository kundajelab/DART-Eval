import os
import sys

from ....evaluators import NTProbingVariantEvaluator
from ....components import VariantDataset
from ....training import CNNEmbeddingsPredictor
import polars as pl


if __name__ == "__main__":
    model_name = "nucleotide-transformer-v2-500m-multi-species"

    batch_size = 32
    num_workers = 4
    seed = 0
    device = "cuda"
    chroms=None
    
    variants_bed = sys.argv[1]
    counts_tsv = sys.argv[2]
    genome_fa = sys.argv[3]
    cell_line = sys.argv[4]

    model_path = f"/scratch/groups/akundaje/dnalm_benchmark/predictors/cell_line_2114/{model_name}/{cell_line}/v3/checkpoint_149.pt"

    out_dir = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/variants/probing/{model_name}/{cell_line}/"
    os.makedirs(out_dir, exist_ok=True)

    # variants_beds = ["/oak/stanford/groups/akundaje/anusri/variant-benchmakring/gm12878.dsqtls.benchmarking.tsv",
    #                  "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/Eu.CaQTLS.tsv",
    #                  "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/Afr.ASB.CaQTLS.tsv", 
    #                  "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/Afr.CaQTLS.tsv"]
    # likelihood_tsvs = ["gm12878.dsqtls.benchmarking.counts.tsv", 
    #                    "Eu.CaQTLS.counts.tsv",
    #                    "Afr.ASB.CaQTLS.counts.tsv", 
    #                    "Afr.CaQTLS.counts.tsv"]
    # genome_fas = ["/oak/stanford/groups/akundaje/soumyak/refs/hg19/male.hg19.fa", 
    #               "/oak/stanford/groups/akundaje/soumyak/refs/hg19/male.hg19.fa",
    #               "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta",
    #               "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"]
    
    input_channels = 1024
    hidden_channels = 32
    kernel_size = 8

    print(counts_tsv)
    out_path = os.path.join(out_dir, f"{counts_tsv}")

    dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
    model = CNNEmbeddingsPredictor(input_channels, hidden_channels, kernel_size)
    evaluator = NTProbingVariantEvaluator(model, model_path, model_name, batch_size, num_workers, device)
    counts_df = evaluator.evaluate(dataset, out_path, progress_bar=True)

    df = dataset.elements_df
    scored_df = pl.concat([df, counts_df], how="horizontal")
    scored_df.write_csv(out_path, separator="\t")
