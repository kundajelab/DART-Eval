import json
import numpy as np
from statsmodels.stats.proportion import proportion_confint
import sys

# Sample JSON file paths for each model (replace these paths with actual paths)

WORK_DIR="/oak/stanford/groups/akundaje/arpitas/dart-eval/"

setting_type = sys.argv[1] if len(sys.argv) > 1 else "probed"

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
        "Spearman r Peaks": data['test_spearman_pos'],
        "Pearson r Peaks": data['test_pearson_pos'],
        "Spearman r All": data['test_spearman_all'],
        "Pearson r All": data['test_pearson_all'],
        "AUROC":  data['test_auroc'],
        "AUPRC": data['test_auprc']
    }
    
    return metrics

def underline_max_values(model_metrics):
    """Underline the maximum values for Accuracy, AUROC, and AUPRC."""
    spearman_pos = {model: metrics["Spearman r Peaks"] for model, metrics in model_metrics.items()}
    pearson_pos = {model: metrics["Pearson r Peaks"] for model, metrics in model_metrics.items()}
    spearman_all = {model: metrics["Spearman r All"] for model, metrics in model_metrics.items()}
    pearson_all = {model: metrics["Pearson r All"] for model, metrics in model_metrics.items()}
    auroc = {model: metrics["AUROC"] for model, metrics in model_metrics.items()}
    auprcs = {model: metrics["AUPRC"] for model, metrics in model_metrics.items()}
    
    # Find the models with maximum values
    max_spearman_pos_model = max(spearman_pos, key=spearman_pos.get)
    max_pearson_pos_model = max(pearson_pos, key=pearson_pos.get)
    max_spearman_all_model = max(spearman_all, key=spearman_all.get)
    max_pearson_all_model = max(pearson_all, key=pearson_all.get)
    max_auroc_model = max(auroc, key=auroc.get)
    max_auprc_model = max(auprcs, key=auprcs.get)
    
    return max_spearman_pos_model, max_pearson_pos_model, max_spearman_all_model, max_pearson_all_model, max_auroc_model, max_auprc_model

def generate_latex_table_per_cell_type(model_metrics, cell_type):
    """Generate a LaTeX table for a specific cell type with the largest values underlined."""
    # Find the models with maximum Accuracy, AUROC, and AUPRC
    max_spearman_pos_model, max_pearson_pos_model, max_spearman_all_model, max_pearson_all_model, max_auroc_model, max_auprc_model = underline_max_values(model_metrics)
    
    latex = f"""
\\begin{{table}}[ht]
\\centering
\\begin{{tabular}}{{|c|c|c|c|}}
\\hline
Model                   & Accuracy & AUROC  & AUPRC  \\\\ \\hline
"""
    
    for model, metrics in model_metrics.items():
    # Add metrics for each model for the specific cell type
        spearman_pos = metrics['Spearman r Peaks']
        pearson_pos = metrics['Pearson r Peaks']
        spearman_all = metrics['Spearman r All']
        pearson_all = metrics['Pearson r All']
        auroc = metrics['AUROC']
        auprc = metrics['AUPRC']
            
            # Underline the largest values
        spearman_pos_str = f"\\underline{{{spearman_pos:.4f}}}" if model == max_spearman_pos_model else f"{spearman_pos:.4f}"
        pearson_pos_str = f"\\underline{{{pearson_pos:.4f}}}" if model == max_pearson_pos_model else f"{pearson_pos:.4f}"
        spearman_all_str = f"\\underline{{{spearman_all:.4f}}}" if model == max_spearman_all_model else f"{spearman_all:.4f}"
        pearson_all_str = f"\\underline{{{pearson_all:.4f}}}" if model == max_pearson_all_model else f"{pearson_all:.4f}"
        auroc_str = f"\\underline{{{auroc:.4f}}}" if model == max_auroc_model else f"{auroc:.4f}"
        auprc_str = f"\\underline{{{auprc:.4f}}}" if model == max_auprc_model else f"{auprc:.4f}"
        
        latex += f"& {model} & {spearman_pos_str} & {pearson_pos_str} & {spearman_all_str} & {pearson_all_str} & {auroc_str} & {auprc_str} \\\\ \n"
    
    latex += f"""\\hline
\\end{{tabular}}
\\caption{{Model performance for {cell_type}.}}
\\end{{table}}
"""
    return latex

def main():
    # Dictionary to store model metrics
    model_metrics = {}
    cell_types = ["GM12878", "H1ESC", "HEPG2", "IMR90", "K562"]

    for cell_type in cell_types:
        model_json_files = {
        "Caduceus": f"{WORK_DIR}/task_4_chromatin_activity/supervised_model_outputs/{setting_type}/caduceus-ps_seqlen-131k_d_model-256_n_layer-16/{cell_type}/eval_test.json",
        "DNABERT-2": f"{WORK_DIR}/task_4_chromatin_activity/supervised_model_outputs/{setting_type}/DNABERT-2-117M/{cell_type}/eval_test.json",
        "GENA-LM": f"{WORK_DIR}/task_4_chromatin_activity/supervised_model_outputs/{setting_type}/gena-lm-bert-large-t2t/{cell_type}/eval_test.json",
        "HyenaDNA": f"{WORK_DIR}/task_4_chromatin_activity/supervised_model_outputs/{setting_type}/hyenadna-large-1m-seqlen-hf/{cell_type}/eval_test.json",
        "Mistral-DNA": f"{WORK_DIR}/task_4_chromatin_activity/supervised_model_outputs/{setting_type}/Mistral-DNA-v1-1.6B-hg38/{cell_type}/eval_test.json",
        "Nucleotide Transformer": f"{WORK_DIR}/task_4_chromatin_activity/supervised_model_outputs/{setting_type}/nucleotide-transformer-v2-500m-multi-species/{cell_type}/eval_test.json",
        # "ChromBPNet": f"{WORK_DIR}/task_4_chromatin_activity/supervised_model_outputs/ab_initio/chrombpnet_like/{cell_type}/eval_test.json"
        }

        # Parse the metrics for each model
        for model_name, json_file in model_json_files.items():
            metrics = parse_model_metrics(json_file)
            model_metrics[model_name] = metrics

        latex_table = generate_latex_table_per_cell_type(model_metrics, cell_type)
        
        # Print or save the LaTeX table for the specific cell type
        print(f"\nLaTeX Table for {cell_type}:\n")
        print(latex_table)
        with open(f'model_comparison_{cell_type}.tex', 'w') as f:
            f.write(latex_table)

if __name__ == "__main__":
    main()