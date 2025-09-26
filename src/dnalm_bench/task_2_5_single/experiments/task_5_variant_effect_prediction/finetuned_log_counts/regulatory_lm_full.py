import torch
import sys
import numpy as np
import json
import os
import polars as pl 

from ....finetune import RegulatoryLMFullFinetuneFullContext
from ....evaluators import FinetunedVariantEvaluator
from ....components import VariantDataset
import pandas as pd
sys.path.append("/users/patelas/regulatory_lm/src/regulatory_lm")
from modeling.model import *
MAPPING = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
model_str_dict = MODULES
FLOAT_DTYPES = {"float32":torch.float32, "float64":torch.float64, "bfloat16":torch.bfloat16, "float16":torch.float16}

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":

    batch_size = 64
    num_workers = 0
    seed = 0
    device = "cuda"
    chroms=None
    
    variants_bed = sys.argv[1]
    base_model_dir = sys.argv[2]
    orig_checkpoint = sys.argv[3]
    finetuned_model_dir = sys.argv[4]
    out_path = sys.argv[5]
    genome_fa = sys.argv[6]
    cell_line = "GM12878"

    emb_channels = 768
    crop = 557
    lora_rank = 8
    lora_alpha = 2 * lora_rank
    lora_dropout = 0.05

    train_log = f"{finetuned_model_dir}/train.log"
    df = pd.read_csv(train_log, sep="\t")
    checkpoint_num = int(df["epoch"][np.argmin(df["val_loss"])])
    model_path = os.path.join(finetuned_model_dir, f"checkpoint_{checkpoint_num}.pt")

    print(orig_checkpoint)
    saved_model_file = os.path.join(base_model_dir, f"checkpoint_{orig_checkpoint}.pt")

    args = json.load(open(os.path.join(base_model_dir, "args.json"), "r"))
    embedder_kwargs = args.get("embedder_kwargs", {})
    encoder_kwargs = args.get("encoder_kwargs", {})
    decoder_kwargs = args.get("decoder_kwargs", {})
    model_kwargs = args.get("model_kwargs", {})

    embedder =  model_str_dict[args["embedder"]](args["embedding_size"], vocab_size=args["num_real_tokens"]+2, masking=True, **embedder_kwargs)


    encoder = model_str_dict[args["encoder"]](args["embedding_size"], args["num_encoder_layers"], **encoder_kwargs)
    decoder = model_str_dict[args["decoder"]](args["embedding_size"], **decoder_kwargs)
    if "classifier" in args:
        classifier = model_str_dict[args["classifier"]](args["embedding_size"])
        model = RegulatoryLMWithClassification(embedder, encoder, decoder, classifier)
    else:
        model = RegulatoryLM(embedder, encoder, decoder)
    model_info = torch.load(saved_model_file)
    if list(model_info["model_state"].keys())[0][:7] == "module.":
        model_info["model_state"] = {x[7:]:model_info["model_state"][x] for x in model_info["model_state"]}
    else:
        model = torch.compile(model)

    model.load_state_dict(model_info["model_state"])
    finetune_model = RegulatoryLMFullFinetuneFullContext(model, 1, emb_channels, seq_input_size=2114)

    checkpoint_resume = torch.load(model_path)
    finetune_model.load_state_dict(checkpoint_resume, strict=False)
    dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
    evaluator = FinetunedVariantEvaluator(finetune_model, batch_size, num_workers, device)
    counts_df = evaluator.evaluate(dataset, out_path, progress_bar=True)

    df = dataset.elements_df
    scored_df = pl.concat([df, counts_df], how="horizontal")
    scored_df.write_csv(out_path, separator="\t")