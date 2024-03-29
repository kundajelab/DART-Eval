import os
import sys

from ....evaluators import DNABERT2VariantEvaluator
from ....components import VariantDataset


if __name__ == "__main__":
    model_name = "DNABERT-2-117M"
    genome_fa = "/oak/stanford/groups/akundaje/soumyak/refs/hg19/male.hg19.fa"
    # variants_bed = "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/Afr.ASB.CaQTLS.tsv"
    variants_bed = "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/gm12878.dsqtls.benchmarking.tsv"
    batch_size = 64
    num_workers = 4
    seed = 0
    device = "cuda"
    chroms=None

    likelihood_tsv = "gm12878.dsqtls.benchmarking_likelihoods.tsv"
    out_dir = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/variants/likelihoods/{model_name}/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{likelihood_tsv}")

    dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
    evaluator = DNABERT2VariantEvaluator(model_name, batch_size, num_workers, device)
    evaluator.evaluate(dataset, out_path, progress_bar=True)
