import os
import sys

from ....evaluators import MistralProbingVariantEvaluator
from ....components import VariantDataset
from ....training import CNNEmbeddingsPredictor


if __name__ == "__main__":
    model_name = "Mistral-DNA-v0.1"

    batch_size = 2048
    num_workers = 4
    seed = 0
    device = "cuda"
    chroms=None
    
    variants_bed = sys.argv[1]
    counts_tsv = sys.argv[2]
    genome_fa = sys.argv[3]
    cell_line = sys.argv[4]

    model_path = f"/scratch/groups/akundaje/dnalm_benchmark/predictors/cell_line_2114/{model_name}/{cell_line}/v3/checkpoint_149.pt"

    out_dir = f"/scratch/groups/akundaje/dnalm_benchmark/likelihoods/variants/probed_models/{model_name}/{cell_line}/"
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
    
    input_channels = 256
    hidden_channels = 32
    kernel_size = 8

    print(counts_tsv)
    out_path = os.path.join(out_dir, f"{counts_tsv}")

    dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
    model = CNNEmbeddingsPredictor(input_channels, hidden_channels, kernel_size)
    evaluator = MistralProbingVariantEvaluator(model, model_path, model_name, batch_size, num_workers, device)
    evaluator.evaluate(dataset, out_path, progress_bar=True)
