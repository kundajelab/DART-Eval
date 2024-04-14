import os
import sys

from .....evaluators import DNABERT2VariantEvaluator
from .....components import VariantDataset


if __name__ == "__main__":
    model_name = "DNABERT-2-117M"
    batch_size = 64
    num_workers = 4
    seed = 0
    device = "cuda"
    chroms=None

    # out_dir = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/variants/likelihoods/{model_name}/"
    out_dir = f"/scratch/groups/akundaje/dnalm_benchmark/likelihoods/variants/{model_name}/"
    os.makedirs(out_dir, exist_ok=True)
    
    variants_beds = [#"/oak/stanford/groups/akundaje/anusri/variant-benchmakring/gm12878.dsqtls.benchmarking.tsv",
                    #  "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/Eu.CaQTLS.tsv",
                    #  "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/Afr.ASB.CaQTLS.tsv", 
                     "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/Afr.CaQTLS.tsv"]
    likelihood_tsvs = [#"gm12878.dsqtls.benchmarking_likelihoods.tsv", 
                       #"Eu.CaQTLS.likelihoods.tsv", 
                       #"Afr.ASB.CaQTLS.likelihoods.tsv", 
                       "Afr.CaQTLS.likelihoods.tsv"]
    genome_fas = [#"/oak/stanford/groups/akundaje/soumyak/refs/hg19/male.hg19.fa", 
                  #"/oak/stanford/groups/akundaje/soumyak/refs/hg19/male.hg19.fa", 
                 #"/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta",
                  "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"]
    for variants_bed, likelihood_tsv, genome_fa in zip(variants_beds, likelihood_tsvs, genome_fas):
        print(likelihood_tsv)
        out_path = os.path.join(out_dir, f"{likelihood_tsv}")

        dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
        evaluator = DNABERT2VariantEvaluator(model_name, batch_size, num_workers, device)
        evaluator.evaluate(dataset, out_path, progress_bar=True)
