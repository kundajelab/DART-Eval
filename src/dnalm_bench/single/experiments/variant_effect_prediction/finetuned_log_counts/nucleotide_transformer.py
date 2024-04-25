import os
import sys

import torch

from ....finetune import NucleotideTransformerLoRAModel
from ....evaluators import FinetunedVariantEvaluator
from ....components import VariantDataset
import polars as pl


if __name__ == "__main__":
    dataset = sys.argv[1] #cell line name

    model_name = "nucleotide-transformer-v2-500m-multi-species"

    genomes = {
        "gm12878.dsqtls": "/home/atwang/dnalm_bench_data/male.hg19.fa", 
        "Eu.CaQTLS": "/home/atwang/dnalm_bench_data/male.hg19.fa",
        "Afr.ASB.CaQTLS": "/home/atwang/dnalm_bench_data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta",
        "Afr.CaQTLS": "/home/atwang/dnalm_bench_data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    }
    genome_fa = genomes[dataset]

    variants_beds = {
        "gm12878.dsqtls": "/home/atwang/dnalm_bench_data/variant-benchmarking/gm12878.dsqtls.benchmarking.tsv", 
        "Eu.CaQTLS": "/home/atwang/dnalm_bench_data/variant-benchmarking/Eu.CaQTLS.tsv", 
        "Afr.ASB.CaQTLS": "/home/atwang/dnalm_bench_data/variant-benchmarking/Afr.ASB.CaQTLS.tsv", 
        "Afr.CaQTLS": "/home/atwang/dnalm_bench_data/variant-benchmarking/Afr.CaQTLS.tsv", 
    }
    variants_bed = variants_beds[dataset]

    cell_line = "GM12878"
    checkpoint_num = 5

    chroms=None

    batch_size = 24
    num_workers = 4
    prefetch_factor = 2
    seed = 0
    device = "cuda"
    emb_channels = 1024

    crop = 557

    lora_rank = 8
    lora_alpha = 2 * lora_rank
    lora_dropout = 0.05

    model_dir = f"/home/atwang/dnalm_bench_data/predictors/cell_line_2114_ft/{model_name}/{cell_line}/v8"
   
    checkpoint_path = os.path.join(model_dir, f"checkpoint_{checkpoint_num}.pt")

    out_dir = f"/home/atwang/dnalm_bench_data/variants/{model_name}/{cell_line}"    
    # out_dir = f"/home/atwang/dnalm_bench_data/predictors/cell_line_2114_ft/{model_name}/{cell_line}/test"    

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{variants_bed}.json")

    model = NucleotideTransformerLoRAModel(model_name, lora_rank, lora_alpha, lora_dropout, 1)
    checkpoint_resume = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_resume, strict=False)
    dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
    evaluator = FinetunedVariantEvaluator(model, model_name, batch_size, num_workers, device)
    counts_df = evaluator.evaluate(dataset, out_path, progress_bar=True)

    df = dataset.elements_df
    scored_df = pl.concat([df, counts_df], how="horizontal")
    scored_df.write_csv(out_path, separator="\t")