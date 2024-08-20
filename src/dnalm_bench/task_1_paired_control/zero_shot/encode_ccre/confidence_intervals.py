import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_confint
import json
import os

root_output_dir = os.environ.get("DART_WORK_DIR", "")

hyena_table = os.path.join(root_output_dir,"task_1_ccre/zero_shot_outputs/likelihoods/hyenadna-large-1m-seqlen-hf/metrics.json")
with open(hyena_table, "r") as f:
    hyena_data = json.load(f)

gena_table = os.path.join(root_output_dir,"task_1_ccre/zero_shot_outputs/likelihoods/gena-lm-bert-large-t2t/metrics.json")
with open(gena_table, "r") as f:
    gena_data = json.load(f)

dnabert_table = os.path.join(root_output_dir,"task_1_ccre/zero_shot_outputs/likelihoods/DNABERT-2-117M/metrics.json")
with open(dnabert_table, "r") as f:
    dnabert_data = json.load(f) 

nt_table = os.path.join(root_output_dir,"task_1_ccre/zero_shot_outputs/likelihoods/nucleotide-transformer-v2-500m-multi-species/metrics.json")
with open(nt_table, "r") as f:
    nt_data = json.load(f)

def get_confidence_interval(data, num_ccres):
    conf_int = proportion_confint(data["acc"]*num_ccres, num_ccres, method="normal")
    interval = conf_int[1] - conf_int[0]
    error = interval / 2
    mean = (conf_int[1] + conf_int[0]) / 2
    return f"{mean:.3f} \pm {error:.3f}"

num_ccres=(2348855-1)*2
print("DNABert2", get_confidence_interval(dnabert_data, num_ccres))
print("Gena LM", get_confidence_interval(gena_data, num_ccres))
print("Hyena DNA", get_confidence_interval(hyena_data, num_ccres))
print("Nuc Transformer", get_confidence_interval(nt_data, num_ccres))

