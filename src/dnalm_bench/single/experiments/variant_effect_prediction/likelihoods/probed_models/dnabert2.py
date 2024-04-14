import os
import sys

from .....evaluators import DNABERT2ProbingVariantEvaluator
from .....components import VariantDataset
from .....training import CNNEmbeddingsPredictor


if __name__ == "__main__":
    model_name = "DNABERT-2-117M"
    
    # out_dir = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/variants/likelihoods/{model_name}/"
    out_dir = f"/scratch/groups/akundaje/dnalm_benchmark/likelihoods/variants/probed_models/{model_name}/"
    os.makedirs(out_dir, exist_ok=True)

    batch_size = 64
    num_workers = 4
    seed = 0
    device = "cuda"
    chroms=None
    
    variants_bed = sys.argv[1]
    likelihood_tsv = sys.argv[2]
    genome_fa = sys.argv[3]
    model_path = sys.argv[4]

    # variants_beds = ["/oak/stanford/groups/akundaje/anusri/variant-benchmakring/gm12878.dsqtls.benchmarking.tsv",
    #                  "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/Eu.CaQTLS.tsv",
    #                  "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/Afr.ASB.CaQTLS.tsv", 
    #                  "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/Afr.CaQTLS.tsv"]
    # likelihood_tsvs = ["gm12878.dsqtls.benchmarking_likelihoods.tsv", 
    #                    "Eu.CaQTLS.likelihoods.tsv",
    #                    "Afr.ASB.CaQTLS.likelihoods.tsv", 
    #                    "Afr.CaQTLS.likelihoods.tsv"]
    # genome_fas = ["/oak/stanford/groups/akundaje/soumyak/refs/hg19/male.hg19.fa", 
    #               "/oak/stanford/groups/akundaje/soumyak/refs/hg19/male.hg19.fa",
    #               "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta",
    #               "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"]
    
    input_channels = 768
    hidden_channels = 32
    kernel_size = 8

    print(likelihood_tsv)
    out_path = os.path.join(out_dir, f"{likelihood_tsv}")

    dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
    model = CNNEmbeddingsPredictor(input_channels, hidden_channels, kernel_size)
    evaluator = DNABERT2ProbingVariantEvaluator(model, model_path, model_name, batch_size, num_workers, device)
    evaluator.evaluate(dataset, out_path, progress_bar=True)