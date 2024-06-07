import os
import sys
import numpy as np
import polars as pl

from ....evaluators import DNABERT2VariantEmbeddingEvaluator
from ....components import VariantDataset

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    model_name = "DNABERT-2-117M"
    batch_size = 512
    num_workers = 0
    seed = 0
    device = "cuda"
    chroms=None

    variants_bed = sys.argv[1]
    output_prefix = sys.argv[2]
    genome_fa = sys.argv[3]
    cell_line = "GM12878"

    out_dir = os.path.join(root_output_dir, f"task_5_variant_effect_prediction/outputs/zero_shot/embeddings/{model_name}/")
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, output_prefix + ".tsv")

    allele1_embeddings_path = os.path.join(out_dir, f"{output_prefix}_allele1_embeddings.npy")
    allele2_embeddings_path = os.path.join(out_dir, f"{output_prefix}_allele2_embeddings.npy")

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
