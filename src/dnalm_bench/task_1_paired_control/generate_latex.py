import json
import numpy as np
from statsmodels.stats.proportion import proportion_confint
import sys

# Sample JSON file paths for each model (replace these paths with actual paths)

WORK_DIR="/oak/stanford/groups/akundaje/arpitas/dart-eval/"

def get_confidence_interval(json_file, num_ccres, metric):
    print("json file", json_file)
    with open(json_file, 'r') as f:
        data = json.load(f)
    if metric == "test_auroc" or metric == "test_auprc":
        return data[metric], f"{data[metric]:.4f}"
    conf_int = proportion_confint(data[metric]*num_ccres, num_ccres, method="normal")
    interval = conf_int[1] - conf_int[0]
    error = interval / 2
    mean = (conf_int[1] + conf_int[0]) / 2
    return mean, f"{mean:.4f} $\pm$ {error:.4e}"

def underline_max_values(model_json_files, num_ccres):
    """Underline the maximum values for Accuracy, AUROC, and AUPRC."""
    print(model_json_files)
    accuracies = dict()
    paired_accuracies = dict()
    aurocs = dict()
    auprcs = dict()
    for model in model_json_files.keys():
        mean, _ = get_confidence_interval(model_json_files[model], num_ccres, "test_acc")
        mean_paired, _ = get_confidence_interval(model_json_files[model], num_ccres, "test_acc_paired")
        accuracies[model] = mean
        paired_accuracies[model] = mean_paired
        aurocs[model] = get_confidence_interval(model_json_files[model], num_ccres, "test_auroc")
        auprcs[model] = get_confidence_interval(model_json_files[model], num_ccres, "test_auprc")
        # Find the models with maximum values
    max_accuracy_model = max(accuracies, key=accuracies.get)
    max_acc_paired_model = max(paired_accuracies, key=paired_accuracies.get)
    max_auroc_model = max(aurocs, key=aurocs.get)
    max_auprc_model = max(auprcs, key=auprcs.get)
    
    return max_accuracy_model, max_acc_paired_model, max_auroc_model, max_auprc_model

def generate_latex_table_per_cell_type(model_json_files, num_ccres):
    """Generate a LaTeX table for a specific cell type with the largest values underlined."""
    # Find the models with maximum Accuracy, AUROC, and AUPRC
    max_values = dict()
    max_values["test_acc"], max_values["test_acc_paired"], max_values["test_auroc"], max_values["test_auprc"] = underline_max_values(model_json_files, num_ccres)
    
    latex = f"""
\\begin{{table}}[ht]
\\centering
\\begin{{tabular}}{{|c|c|c|c|}}
\\hline
Model & Accuracy \\\\ \\hline
"""
    # Add metrics for each model for the specific cell type
    for model in model_json_files.keys():
        latex += f"& {model} " 
        for metric in ["test_acc", "test_acc_paired", "test_auroc", "test_auprc"]:
            _, latex_model = get_confidence_interval(model_json_files[model], num_ccres, metric)
            metric_str = f"\\underline{{{latex_model}}}" if model == max_values[metric] else f"{latex_model}"
            latex += f"& {metric_str} "
        latex += "\\\\ \n"
        # Underline the largest values
        # acc_str = f"\\underline{{{latex_model}}}" if model == max_acc_model else f"{latex_model}"

        # paired_acc_str = f"\\underline{{{latex_model}}}" if model == max_acc_paired_model else f"{latex_model}"
        # auroc_str = f"\\underline{{{latex_model}}}" if model == max_auroc_model else f"{latex_model}"
        # auprc_str = f"\\underline{{{latex_model}}}" if model == max_auprc_model else f"{latex_model}"
        
        # latex += f"& {model} & {acc_str} & {paired_acc_str} & {auroc_str} & {auprc_str} \\\\ \n"
    
    latex += f"""\\hline
\\end{{tabular}}
\\end{{table}}
"""
    return latex

def main():
    setting_type = sys.argv[1] if len(sys.argv) > 1 else "probed"
    model_json_files = {
        "DNABERT-2": f"{WORK_DIR}/task_1_ccre/supervised_model_outputs/{setting_type}/DNABERT-2-117M/eval_test.json",
        "GENA-LM": f"{WORK_DIR}/task_1_ccre/supervised_model_outputs/{setting_type}/gena-lm-bert-large-t2t/eval_test.json",
        "HyenaDNA": f"{WORK_DIR}/task_1_ccre/supervised_model_outputs/{setting_type}/hyenadna-large-1m-seqlen-hf/eval_test.json",
        "Nucleotide Transformer": f"{WORK_DIR}/task_1_ccre/supervised_model_outputs/{setting_type}/nucleotide-transformer-v2-500m-multi-species/eval_test.json",
        "Caduceus": f"{WORK_DIR}/task_1_ccre/supervised_model_outputs/{setting_type}/caduceus-ps_seqlen-131k_d_model-256_n_layer-16/eval_test.json",
        "Mistral-DNA": f"{WORK_DIR}/task_1_ccre/supervised_model_outputs/{setting_type}/Mistral-DNA-v1-1.6B-hg38/eval_test.json"
    }

    num_ccres=(2348855-1)*2

    latex_table = generate_latex_table_per_cell_type(model_json_files, num_ccres)
    
    # Print or save the LaTeX table for the specific cell type
    print(f"\nLaTeX Table:\n")
    print(latex_table)

if __name__ == "__main__":
    main()