import os
import sys
import polars as pl

from ....evaluators import NTVariantSingleTokenEvaluator
from ....components import VariantDataset

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    dataset = sys.argv[1]

    model_name = "nucleotide-transformer-v2-500m-multi-species"
    batch_size = 128
    num_workers = 0
    seed = 0
    device = "cuda"
    chroms=None

    variants_bed = sys.argv[1]
    output_prefix = sys.argv[2]
    genome_fa = sys.argv[3]
    cell_line = "GM12878"

    out_dir = os.path.join(root_output_dir, f"task_5_variant_effect_prediction/outputs/zero_shot/likelihoods/{model_name}/")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{output_prefix}.tsv")

    dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
    evaluator = NTVariantSingleTokenEvaluator(model_name, batch_size, num_workers, device)    
    score_df = evaluator.evaluate(dataset, out_path, progress_bar=True)

    df = dataset.elements_df
    scored_df = pl.concat([df, score_df], how="horizontal")
    print(out_path)
    scored_df.write_csv(out_path, separator="\t")
