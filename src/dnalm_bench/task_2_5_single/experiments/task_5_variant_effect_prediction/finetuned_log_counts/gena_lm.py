import os
import sys

import torch

from ....finetune import GENALMLoRAModel
from ....evaluators import FinetunedVariantEvaluator
from ....components import VariantDataset
import polars as pl
import pandas as pd
import numpy as np

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    model_name = "gena-lm-bert-large-t2t"
    batch_size = 512
    num_workers = 0
    seed = 0
    device = "cuda"
    chroms=None
    
    variants_bed = sys.argv[1]
    output_prefix = sys.argv[2]
    genome_fa = sys.argv[3]
    cell_line = "GM12878"

    emb_channels = 768
    crop = 557
    lora_rank = 8
    lora_alpha = 2 * lora_rank
    lora_dropout = 0.05

    model_folder = os.path.join(root_output_dir, f"task_4_chromatin_activity/supervised_models/fine_tuned/{model_name}/{cell_line}")
    train_log = f"{model_folder}/train.log"
    df = pd.read_csv(train_log, sep="\t")
    checkpoint_num = int(df["epoch"][np.argmin(df["val_loss"])])
    model_path = os.path.join(model_folder, f"checkpoint_{checkpoint_num}.pt")

    out_dir = os.path.join(root_output_dir, f"task_5_variant_effect_prediction/outputs/fine_tuned/{model_name}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{output_prefix}.tsv")

    model = GENALMLoRAModel(model_name, lora_rank, lora_alpha, lora_dropout, 1)
    checkpoint_resume = torch.load(model_path)
    model.load_state_dict(checkpoint_resume, strict=False)
    dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
    evaluator = FinetunedVariantEvaluator(model, batch_size, num_workers, device)
    counts_df = evaluator.evaluate(dataset, out_path, progress_bar=True)

    df = dataset.elements_df
    scored_df = pl.concat([df, counts_df], how="horizontal")
    scored_df.write_csv(out_path, separator="\t")