import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_confint
import json
import os

root_output_dir = os.environ.get("DART_WORK_DIR", "")

hyena_table_probed = os.path.join(root_output_dir,"task_3_cell-type-specific/supervised_model_outputs/probed/hyenadna-large-1m-seqlen-hf/eval_test.json")
with open(hyena_table_probed, "r") as f:
    hyena_probed = json.load(f)

hyena_table_finetuned = os.path.join(root_output_dir,"task_3_cell-type-specific/supervised_model_outputs/fine_tuned/hyenadna-large-1m-seqlen-hf/eval_test.json")
with open(hyena_table_finetuned, "r") as f:
    hyena_finetuned = json.load(f)

gena_table_probed = os.path.join(root_output_dir,"task_3_cell-type-specific/supervised_model_outputs/probed/gena-lm-bert-large-t2t/eval_test.json")
with open(gena_table_probed, "r") as f:
    gena_probed = json.load(f)

gena_table_finetuned = os.path.join(root_output_dir,"task_3_cell-type-specific/supervised_model_outputs/fine_tuned/gena-lm-bert-large-t2t/eval_test.json")
with open(gena_table_finetuned, "r") as f:
    gena_finetuned = json.load(f)

dnabert_table_probed = os.path.join(root_output_dir,"task_3_cell-type-specific/supervised_model_outputs/probed/DNABERT-2-117M/eval_test.json")
with open(dnabert_table_probed, "r") as f:
    dnabert_probed = json.load(f) 

dnabert_table_finetuned = os.path.join(root_output_dir,"task_3_cell-type-specific/supervised_model_outputs/fine_tuned/DNABERT-2-117M/eval_test.json")
with open(dnabert_table_finetuned, "r") as f:
    dnabert_finetuned = json.load(f)

nt_table_probed = os.path.join(root_output_dir,"task_3_cell-type-specific/supervised_model_outputs/probed/nucleotide-transformer-v2-500m-multi-species/eval_test.json")
with open(nt_table_probed, "r") as f:
    nt_probed = json.load(f)

nt_table_finetuned = os.path.join(root_output_dir,"task_3_cell-type-specific/supervised_model_outputs/fine_tuned/nucleotide-transformer-v2-500m-multi-species/eval_test.json")
with open(nt_table_finetuned, "r") as f:
    nt_finetuned = json.load(f)

def get_confidence_interval(data, num_ccres):
    conf_int = proportion_confint(data["test_acc"]*num_ccres, num_ccres, method="normal")
    interval = conf_int[1] - conf_int[0]
    error = interval / 2
    mean = (conf_int[1] + conf_int[0]) / 2
    return f"{mean:.3f} \pm {error:.3f}"

num_ccres=(216747-1)
print("PROBED")
print("DNABert2", get_confidence_interval(dnabert_probed, num_ccres))
print("Gena LM", get_confidence_interval(gena_probed, num_ccres))
print("Hyena DNA", get_confidence_interval(hyena_probed, num_ccres))
print("Nucleotide Transformer", get_confidence_interval(nt_probed, num_ccres))

print("\nFINETUNED")
print("DNABert2", get_confidence_interval(dnabert_finetuned, num_ccres))
print("Gena LM", get_confidence_interval(gena_finetuned, num_ccres))
print("Hyena DNA", get_confidence_interval(hyena_finetuned, num_ccres))
print("Nucleotide Transformer", get_confidence_interval(nt_finetuned, num_ccres))