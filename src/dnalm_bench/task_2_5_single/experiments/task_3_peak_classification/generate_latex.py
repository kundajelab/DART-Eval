import json
import numpy as np
from statsmodels.stats.proportion import proportion_confint
import sys

# Sample JSON file paths for each model (replace these paths with actual paths)

WORK_DIR="/oak/stanford/groups/akundaje/arpitas/dart-eval/"

setting_type = sys.argv[1] if len(sys.argv) > 1 else "probed"
model_json_files = {
    "Caduceus": f"{WORK_DIR}/task_3_peak_classification/supervised_model_outputs/{setting_type}/caduceus-ps_seqlen-131k_d_model-256_n_layer-16/eval_test.json",
    "DNABERT-2": f"{WORK_DIR}/task_3_peak_classification/supervised_model_outputs/{setting_type}/DNABERT-2-117M/eval_test.json",
    "GENA-LM": f"{WORK_DIR}/task_3_peak_classification/supervised_model_outputs/{setting_type}/gena-lm-bert-large-t2t/eval_test.json",
    "HyenaDNA": f"{WORK_DIR}/task_3_peak_classification/supervised_model_outputs/{setting_type}/hyenadna-large-1m-seqlen-hf/eval_test.json",
    "Mistral-DNA": f"{WORK_DIR}/task_3_peak_classification/supervised_model_outputs/{setting_type}/Mistral-DNA-v1-1.6B-hg38/eval_test.json",
    "Nucleotide Transformer": f"{WORK_DIR}/task_3_peak_classification/supervised_model_outputs/{setting_type}/nucleotide-transformer-v2-500m-multi-species/eval_test.json",
    "Probing-head-like": f"{WORK_DIR}/task_3_peak_classification/supervised_model_outputs/ab_initio/probing_head_like/eval_test.json",
    "ChromBPNet-like": f"{WORK_DIR}/task_3_peak_classification/supervised_model_outputs/ab_initio/chrombpnet_like/eval_test.json"
}

num_peaks=(216747-1)
def get_confidence_interval(acc):
    conf_int = proportion_confint(acc*num_peaks, num_peaks, method="normal")
    interval = conf_int[1] - conf_int[0]
    error = interval / 2
    mean = (conf_int[1] + conf_int[0]) / 2
    return mean, f"{mean:.4f} $\pm$ {error:.4e}"

def parse_model_metrics(json_file):
    """Parse the model's metrics from the JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract accuracy, AUROC, and AUPRC for each class
    metrics = {
        "GM12878": {
            "Accuracy": get_confidence_interval(data['class_GM12878_acc'])[0],
            "Accuracy_Latex": get_confidence_interval(data['class_GM12878_acc'])[1],
            "AUROC": data['class_GM12878_auroc'],
            "AUPRC": data['class_GM12878_auprc']
        },
        "H1ESC": {
            "Accuracy": get_confidence_interval(data['class_H1ESC_acc'])[0],
            "Accuracy_Latex": get_confidence_interval(data['class_H1ESC_acc'])[1],
            "AUROC": data['class_H1ESC_auroc'],
            "AUPRC": data['class_H1ESC_auprc']
        },
        "HEPG2": {
            "Accuracy": get_confidence_interval(data['class_HEPG2_acc'])[0],
            "Accuracy_Latex": get_confidence_interval(data['class_HEPG2_acc'])[1],
            "AUROC": data['class_HEPG2_auroc'],
            "AUPRC": data['class_HEPG2_auprc']
        },
        "IMR90": {
            "Accuracy": get_confidence_interval(data['class_IMR90_acc'])[0],
            "Accuracy_Latex": get_confidence_interval(data['class_IMR90_acc'])[1],
            "AUROC": data['class_IMR90_auroc'],
            "AUPRC": data['class_IMR90_auprc']
        },
        "K562": {
            "Accuracy": get_confidence_interval(data['class_K562_acc'])[0],
            "Accuracy_Latex": get_confidence_interval(data['class_K562_acc'])[1],
            "AUROC": data['class_K562_auroc'],
            "AUPRC": data['class_K562_auprc']
        }
    }
    
    return metrics

def underline_max_values(model_metrics, cell_type):
    """Underline the maximum values for Accuracy, AUROC, and AUPRC."""
    accuracies = {model: metrics[cell_type]["Accuracy"] for model, metrics in model_metrics.items()}
    aurocs = {model: metrics[cell_type]["AUROC"] for model, metrics in model_metrics.items()}
    auprcs = {model: metrics[cell_type]["AUPRC"] for model, metrics in model_metrics.items()}
    
    # Find the models with maximum values
    max_accuracy_model = max(accuracies, key=accuracies.get)
    max_auroc_model = max(aurocs, key=aurocs.get)
    max_auprc_model = max(auprcs, key=auprcs.get)
    
    return max_accuracy_model, max_auroc_model, max_auprc_model

def generate_latex_table_per_cell_type(model_metrics, cell_type):
    """Generate a LaTeX table for a specific cell type with the largest values underlined."""
    # Find the models with maximum Accuracy, AUROC, and AUPRC
    max_acc_model, max_auroc_model, max_auprc_model = underline_max_values(model_metrics, cell_type)
    
    latex = f"""
\\begin{{table}}[ht]
\\centering
\\begin{{tabular}}{{|c|c|c|c|}}
\\hline
Model                   & Accuracy & AUROC  & AUPRC  \\\\ \\hline
"""
    # Add metrics for each model for the specific cell type
    for model, cell_types in model_metrics.items():
        metrics = cell_types[cell_type]
        accuracy = metrics['Accuracy']
        accuracy_latex = metrics['Accuracy_Latex']
        auroc = metrics['AUROC']
        auprc = metrics['AUPRC']
        
        # Underline the largest values
        acc_str = f"\\underline{{{accuracy_latex}}}" if model == max_acc_model else f"{accuracy_latex}"
        auroc_str = f"\\underline{{{auroc:.4f}}}" if model == max_auroc_model else f"{auroc:.4f}"
        auprc_str = f"\\underline{{{auprc:.4f}}}" if model == max_auprc_model else f"{auprc:.4f}"
        
        latex += f"& {model} & {acc_str} & {auroc_str} & {auprc_str} \\\\ \n"
    
    latex += f"""\\hline
\\end{{tabular}}
\\caption{{Model performance for {cell_type}.}}
\\end{{table}}
"""
    return latex

def main():
    # Dictionary to store model metrics
    model_metrics = {}

    # Parse the metrics for each model
    for model_name, json_file in model_json_files.items():
        metrics = parse_model_metrics(json_file)
        model_metrics[model_name] = metrics

    # Generate and print a LaTeX table for each cell type
    cell_types = ["GM12878", "H1ESC", "HEPG2", "IMR90", "K562"]
    
    for cell_type in cell_types:
        latex_table = generate_latex_table_per_cell_type(model_metrics, cell_type)
        
        # Print or save the LaTeX table for the specific cell type
        print(f"\nLaTeX Table for {cell_type}:\n")
        print(latex_table)
        with open(f'model_comparison_{cell_type}.tex', 'w') as f:
            f.write(latex_table)

if __name__ == "__main__":
    main()