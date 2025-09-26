import os
import sys
import json
import torch
import numpy as np
import pandas as pd

from ....finetune import ChromatinEndToEndDataset, evaluate_finetuned_chromatin_model, RegulatoryLMLoRAModel
sys.path.append("/users/patelas/regulatory_lm/src/regulatory_lm")
from modeling.model import *
MAPPING = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
model_str_dict = MODULES
FLOAT_DTYPES = {"float32":torch.float32, "float64":torch.float64, "bfloat16":torch.bfloat16, "float16":torch.float16}

work_dir = os.environ.get("DART_WORK_DIR", "")
cache_dir = os.environ.get("DART_CACHE_DIR")

if __name__ == "__main__":
    cell_line = sys.argv[1] #cell line name


    genome_fa = "/mnt/lab_data2/regulatory_lm/oak_backup/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"

    peaks_tsv = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_peaks.bed")
    idr_peaks_tsv = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/cell_line_idr_peaks/{cell_line}.bed")
    nonpeaks_tsv = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_nonpeaks.bed")
    assay_bw = os.path.join(work_dir, f"task_4_chromatin_activity/processed_data/bigwigs/{cell_line}_unstranded.bw")

    batch_size = 16
    num_workers = 4
    prefetch_factor = 2
    seed = 0
    device = "cuda"

    chroms_train = [
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr7",
        "chr8",
        "chr9",
        "chr11",
        "chr12",
        "chr13",
        "chr15",
        "chr16",
        "chr17",
        "chr19",
        "chrX",
        "chrY"
    ]
    
    chroms_val = [
        "chr6",
        "chr21"
    ]

    chroms_test = [
        "chr5",
        "chr10",
        "chr14",
        "chr18",
        "chr20",
        "chr22"
    ]
    modes = {"train": chroms_train, "val": chroms_val, "test": chroms_test}
    eval_mode = "test"

    emb_channels = 1024

    crop = 557

    lora_rank = 8
    lora_alpha = 2 * lora_rank
    lora_dropout = 0.05

    accumulate = 8
    
    pos_dataset = ChromatinEndToEndDataset(genome_fa, assay_bw, peaks_tsv, modes[eval_mode], crop, cache_dir=cache_dir)
    idr_dataset = ChromatinEndToEndDataset(genome_fa, assay_bw, idr_peaks_tsv, modes[eval_mode], crop, cache_dir=cache_dir)
    neg_dataset = ChromatinEndToEndDataset(genome_fa, assay_bw, nonpeaks_tsv, modes[eval_mode], crop, cache_dir=cache_dir)

    model_dir = sys.argv[2]    
    train_log = f"{model_dir}/train.log"
    df = pd.read_csv(train_log, sep="\t")
    checkpoint_num = int(df["epoch"][np.argmin(df["val_loss"])])    
    print(checkpoint_num)
    checkpoint_path = os.path.join(model_dir, f"checkpoint_{checkpoint_num}.pt")


    base_model_dir = sys.argv[3]
    orig_checkpoint = sys.argv[4]
    out_path = sys.argv[5]
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
    lora_model = RegulatoryLMLoRAModel(model, lora_rank, lora_alpha, lora_dropout, 1, emb_channels, seq_input_size=2114, model_input_size=2114)

    checkpoint_resume = torch.load(checkpoint_path)
    lora_model.load_state_dict(checkpoint_resume, strict=False)

    metrics = evaluate_finetuned_chromatin_model(pos_dataset, idr_dataset, neg_dataset, lora_model, batch_size, out_path,
                                       num_workers, prefetch_factor, device, progress_bar=True)
    
    for k, v in metrics.items():
        print(f"{k}: {v}")