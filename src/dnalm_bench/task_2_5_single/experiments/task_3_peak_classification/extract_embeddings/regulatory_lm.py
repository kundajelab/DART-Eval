import torch
import sys
import numpy as np
import json
import os
from ....components import SimpleSequence
from ....embeddings import RegulatoryLMEmbeddingExtractor

sys.path.append("/users/patelas/regulatory_lm/src/regulatory_lm")
from modeling.model import *
MAPPING = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
model_str_dict = MODULES
FLOAT_DTYPES = {"float32":torch.float32, "float64":torch.float64, "bfloat16":torch.bfloat16, "float16":torch.float16}

root_output_dir = os.environ.get("DART_WORK_DIR", "")


if __name__ == "__main__":
    model_dir = sys.argv[1]
    checkpoint = sys.argv[2]
    out_dir = sys.argv[3]

    seq_len = 500
    model_len = 350
    chroms = None
    batch_size = 64
    num_workers = 0
    seed = 0
    device = "cuda"
    genome_fa = os.path.join(root_output_dir,"refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    elements_tsv = os.path.join(root_output_dir,"task_3_peak_classification/processed_inputs/peaks_by_cell_label_unique_dataloader_format.tsv")


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

    dataset = SimpleSequence(genome_fa, elements_tsv, chroms, seed)
    extractor = RegulatoryLMEmbeddingExtractor(model, batch_size, num_workers, device, seq_input_size=seq_len)
    extractor.extract_embeddings(dataset, os.path.join(out_dir, "task3_embeddings.h5"), progress_bar=True)
