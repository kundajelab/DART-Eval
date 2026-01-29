import torch
import sys
import numpy as np
import json
import os
import polars as pl 

from ....evaluators import RegulatoryLMVariantSingleTokenEvaluator
from ....components import VariantDataset
arsenal_dir = os.environ.get("ARSENAL_MODEL_DIR", "")
sys.path.append(f"{arsenal_dir}/src/regulatory_lm/")
from modeling.model import *
MAPPING = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
model_str_dict = MODULES
FLOAT_DTYPES = {"float32":torch.float32, "float64":torch.float64, "bfloat16":torch.bfloat16, "float16":torch.float16}

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    # dataset = sys.argv[1]

    batch_size = 256
    num_workers = 0
    seed = 0
    device = "cuda"
    chroms=None

    variants_bed = sys.argv[1]
    model_dir = sys.argv[2]
    checkpoint = sys.argv[3]
    out_path = sys.argv[4]
    genome_fa = sys.argv[5] #"/mnt/lab_data2/regulatory_lm/oak_backup/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    cell_line = "GM12878"

    dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)

    model_dir = model_dir + "/" if model_dir[-1] != "/" else model_dir
    args = json.load(open(os.path.join(model_dir, "args.json"), "r"))
    saved_model_file = os.path.join(model_dir, f"checkpoint_{checkpoint}.pt")

    embedder_kwargs = args.get("embedder_kwargs", {})
    encoder_kwargs = args.get("encoder_kwargs", {})
    decoder_kwargs = args.get("decoder_kwargs", {})
    model_kwargs = args.get("model_kwargs", {})

    embedder =  model_str_dict[args["embedder"]](args["embedding_size"], vocab_size=args["num_real_tokens"]+2, masking=True, **embedder_kwargs)


    encoder = model_str_dict[args["encoder"]](args["embedding_size"], args["num_encoder_layers"], **encoder_kwargs)
    decoder = model_str_dict[args["decoder"]](args["embedding_size"], **decoder_kwargs)
    if args["model"] == "RegulatoryLMWithClassification":
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

    model.eval()

    evaluator = RegulatoryLMVariantSingleTokenEvaluator(model, batch_size, num_workers, device, category=0)
    score_df = evaluator.evaluate(dataset, out_path, progress_bar=True)

    df = dataset.elements_df
    scored_df = pl.concat([df, score_df], how="horizontal")
    print(out_path)
    scored_df.write_csv(out_path, separator="\t")
