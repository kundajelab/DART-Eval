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

work_dir = os.environ.get("DART_WORK_DIR", "")


if __name__ == "__main__":
    model_dir = sys.argv[1]
    checkpoint = sys.argv[2]
    out_dir = sys.argv[3]
    cell_line = sys.argv[4]
    category = sys.argv[5]

    chroms = None
    batch_size = 64
    num_workers = 0
    seed = 0
    device = "cuda"

    if category == "idr":
        elements_tsv = os.path.join(root_output_dir, f"task_4_chromatin_activity/processed_data/cell_line_idr_peaks/{cell_line}.bed")
    else:
        elements_tsv = os.path.join(root_output_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_{category}.bed")
    chroms = None
    batch_size = 256
    num_workers = 0
    seed = 0
    device = "cuda"

    model_dir = model_dir + "/" if model_dir[-1] != "/" else model_dir
    args = json.load(open(os.path.join(model_dir, "args.json"), "r"))
    saved_model_file = os.path.join(model_dir, f"checkpoint_{checkpoint}.pt")
    if args["embedder"] == "InputSeqOnlyEmbedder":
        embedder = model_str_dict[args["embedder"]](args["embedding_size"], vocab_size=args["num_real_tokens"]+2, masking=True)
    elif args["embedder"] == "InputBertSeqOnlyEmbedder":
        embedder = model_str_dict[args["embedder"]](args["embedding_size"], seq_len=args["embedder_kwargs"]["seq_len"], vocab_size=args["num_real_tokens"]+2, masking=True)
    else:
        embedder = model_str_dict[args["embedder"]](args["embedding_size"], args["num_categories"], vocab_size=args["num_real_tokens"]+2, masking=True)

    encoder = model_str_dict[args["encoder"]](args["embedding_size"], args["num_encoder_layers"], n_filters=args["embedding_size"])
    decoder = model_str_dict[args["decoder"]](args["embedding_size"])
    if "classifier" in args:
        classifier = model_str_dict[args["classifier"]](args["embedding_size"])
        model = RegulatoryLMWithClassification(embedder, encoder, decoder, classifier)
    else:
        model = RegulatoryLM(embedder, encoder, decoder)
    model = torch.compile(model)
    model_info = torch.load(saved_model_file)
    model.load_state_dict(model_info["model_state"])

    extractor = RegulatoryLMEmbeddingExtractor(model, batch_size, num_workers, device)
    dataset = SimpleSequence(genome_fa, elements_tsv, chroms, seed)
    extractor.extract_embeddings(dataset, os.path.join(out_dir, f"task4_embeddings_{cell_line}_{category}.h5"), progress_bar=True)