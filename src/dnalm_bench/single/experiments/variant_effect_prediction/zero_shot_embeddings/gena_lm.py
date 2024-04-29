import os
import sys
import numpy as np

from ....evaluators import GenaLMVariantEmbeddingEvaluator
from ....components import VariantDataset


if __name__ == "__main__":
    dataset = sys.argv[1]

    model_name = "gena-lm-bert-large-t2t"

    genomes = {
        "gm12878.dsqtls.benchmarking": "/home/atwang/dnalm_bench_data/male.hg19.fa", 
        "Eu.CaQTLS": "/home/atwang/dnalm_bench_data/male.hg19.fa",
        "Afr.ASB.CaQTLS": "/home/atwang/dnalm_bench_data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta",
        "Afr.CaQTLS": "/home/atwang/dnalm_bench_data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    }
    genome_fa = genomes[dataset]

    variants_bed = f"/home/atwang/dnalm_bench_data/variant-benchmarking/{dataset}.tsv" 

    # variants_bed = sys.argv[1]
    # likelihood_tsv = sys.argv[2]
    # genome_fa = sys.argv[3]

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
    
    
    batch_size = 256
    num_workers = 4
    seed = 0
    device = "cuda"
    chroms=None
    # out_dir = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/variants/likelihoods/{model_name}/"
    # out_dir = f"/home/atwang/dnalm_bench_data/likelihoods/variants/{model_name}/"
    evaluator = GenaLMVariantEmbeddingEvaluator(model_name, batch_size, num_workers, device)

    # out_dir = f"/scratch/groups/akundaje/dnalm_benchmark/likelihoods/variants/{model_name}/"
    out_dir = f"/home/atwang/dnalm_bench_data/embeddings/variants/{model_name}/"
    os.makedirs(out_dir, exist_ok=True)

    # for variants_bed, likelihood_tsv, genome_fa in zip(variants_beds, likelihood_tsvs, genome_fas):
    # print(likelihood_tsv)

    out_path = os.path.join(out_dir, f"{dataset}.tsv")
    dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)  
    _, allele1_embeddings, allele2_embeddings = evaluator.evaluate(dataset, out_path, progress_bar=True)

    # Save embeddings
    allele1_embeddings_path = os.path.join(out_dir, f"{dataset}_allele1_embeddings.npy")
    allele2_embeddings_path = os.path.join(out_dir, f"{dataset}_allele2_embeddings.npy")
    np.save(allele1_embeddings_path, allele1_embeddings)
    np.save(allele2_embeddings_path, allele2_embeddings)

    