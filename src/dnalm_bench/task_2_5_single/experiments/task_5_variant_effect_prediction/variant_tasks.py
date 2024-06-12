import numpy as np
import os
from sklearn.metrics.pairwise import manhattan_distances
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

work_dir = os.environ.get("DART_WORK_DIR", "")
work_dir = "/scratch/groups/akundaje/chrombench/synapse"

def get_precision_recall_auc(ctrl_counts, sig_counts):
    counts = np.concatenate([ctrl_counts, sig_counts])
    labels = np.concatenate([np.zeros(len(ctrl_counts)), np.ones(len(sig_counts))])
    auroc = roc_auc_score(labels, counts)
    auprc = average_precision_score(labels, counts)
    return auprc, auroc

def compute_precision_recall(labels, llm_scores):
    labels = np.array(labels)
    llm_scores = np.array(llm_scores)
    
    sorted_indices = np.argsort(llm_scores)[::-1]
    sorted_indices = sorted_indices.astype(int) 
    llm_scores = llm_scores[sorted_indices]
    labels = labels[sorted_indices]

    precisions = []
    recalls = []
    thresholds = []

    tp = 0
    fp = 0
    fn = np.sum(labels)

    for i in range(len(labels)):
        if labels[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(llm_scores[i])

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    return precisions, recalls

def compute_change(filtered_variants_df, switch=False):
    if switch:
        filtered_variants_df["llm_logfc"] = filtered_variants_df["allele1_scores"]-filtered_variants_df["allele2_scores"]
    else:
        filtered_variants_df["llm_logfc"] = filtered_variants_df["allele2_scores"]-filtered_variants_df["allele1_scores"]
    return filtered_variants_df, np.abs(filtered_variants_df["llm_logfc"])

def sig_ctrl_variants_Afr_CaQTLs(scores_data_path):
    afr_caqtls_data_path =  os.path.join(work_dir, "task_5_variant_effect_prediction/input_data/Afr.CaQTLS.tsv")
    afr_caQTLs_df = pd.read_csv(afr_caqtls_data_path, sep="\t")
    likelihoods = pd.read_csv(scores_data_path, sep="\t")

    if "allele1_scores" in likelihoods.columns:
        columns = ["allele1_scores", "allele2_scores"]
    elif "cosine_distance" in likelihoods.columns:
        columns = ["cosine_distance"]
    else:
        columns = []

    scores = likelihoods[["chr_hg38", "pos_hg38", "allele1", "allele2"] + columns]
    if afr_caQTLs_df.shape[0] == scores.shape[0]:
        likelihoods_data = pd.merge(afr_caQTLs_df, scores, on=["chr_hg38", "pos_hg38", "allele1", "allele2"])
        filtered_var_afr_caQTLs_df = likelihoods_data[(likelihoods_data["IsUsed"]==True) & (likelihoods_data["in_peaks"]==True)].copy(deep=True)

        print("unique label values", filtered_var_afr_caQTLs_df["label"].value_counts())

        filtered_var_afrcaqtls_df_sig = filtered_var_afr_caQTLs_df[(filtered_var_afr_caQTLs_df["label"]==1)]
        filtered_var_afrcaqtls_df_ctrl = filtered_var_afr_caQTLs_df[(filtered_var_afr_caQTLs_df["label"]==0)]

        return filtered_var_afrcaqtls_df_ctrl, filtered_var_afrcaqtls_df_sig
    
def variants_Yoruba_LCL_dsQTLs(scores_data_path):
    yoruba_dsqtls_data_path = os.path.join(work_dir, "task_5_variant_effect_prediction/input_data/yoruban.dsqtls.benchmarking.tsv")
    yoruba_dsQTLs_df = pd.read_csv(yoruba_dsqtls_data_path, sep="\t")
    likelihoods = pd.read_csv(scores_data_path, sep="\t")

    if "allele1_scores" in likelihoods.columns:
        columns = ["allele1_scores", "allele2_scores"]
    elif "cosine_distance" in likelihoods.columns:
        columns = ["cosine_distance"]
    else:
        columns = []

    scores = likelihoods[["var.chrom", "var.pos", "var.allele1", "var.allele2"]+columns]
    if yoruba_dsQTLs_df.shape[0] == scores.shape[0]:
        likelihoods_data = pd.merge(yoruba_dsQTLs_df, scores, on=["var.chrom", "var.pos", "var.allele1", "var.allele2"])
        filtered_var_yoruba_dsQTLs_df = likelihoods_data[likelihoods_data["var.isused"]==True].copy(deep=True)

        print("unique label values", filtered_var_yoruba_dsQTLs_df["var.label"].value_counts())

        filtered_var_yoruba_dsqtls_df_sig = filtered_var_yoruba_dsQTLs_df[(filtered_var_yoruba_dsQTLs_df["var.label"]==1)]
        filtered_var_yoruba_dsqtls_df_ctrl = filtered_var_yoruba_dsQTLs_df[(filtered_var_yoruba_dsQTLs_df["var.label"]==-1)]
        
        return filtered_var_yoruba_dsqtls_df_ctrl, filtered_var_yoruba_dsqtls_df_sig
    
def sig_ctrl_variants_Afr_CaQTLs_probed_counts(counts_data_path):
    
    counts_data = pd.read_csv(counts_data_path, sep="\t")
    print(counts_data.columns)
    filtered_var_afr_caQTLs_df = counts_data[counts_data["IsUsed"]==True].copy(deep=True)
    filtered_var_afr_caQTLs_df["llm_logfc"] = filtered_var_afr_caQTLs_df["allele2_scores"] - filtered_var_afr_caQTLs_df["allele1_scores"]

    print("unique label values", np.unique(filtered_var_afr_caQTLs_df["label"]))
    filtered_var_afrcaqtls_df_sig = filtered_var_afr_caQTLs_df[(filtered_var_afr_caQTLs_df["label"]==1) & (-np.log10(filtered_var_afr_caQTLs_df["pval"])>=3)]
    filtered_var_afrcaqtls_df_ctrl = filtered_var_afr_caQTLs_df[(filtered_var_afr_caQTLs_df["label"]==0) & (-np.log10(filtered_var_afr_caQTLs_df["pval"])<3)]
    
    control_counts = np.abs(filtered_var_afrcaqtls_df_ctrl["llm_logfc"])
    sig_counts = np.abs(filtered_var_afrcaqtls_df_sig["llm_logfc"])  

    print(len(control_counts), len(sig_counts))

    counts_ctrl, bins_ctrl = np.histogram(control_counts, bins=100)

    counts_sig, bins_sig = np.histogram(sig_counts, bins=100)

    U1, p = mannwhitneyu(counts_ctrl, counts_sig, alternative="greater")

    return control_counts, sig_counts, filtered_var_afr_caQTLs_df

def beta_logfc(filtered_df, title, ylabel="LogFC Scores", yaxis="llm_logfc"):
    if "Beta" in filtered_df.columns:
        x = filtered_df["Beta"]
    else:
        x = filtered_df["beta"]
    y = filtered_df[yaxis]

    g = sns.jointplot(x=x, y=y, 
                    kind="scatter")
    
    g.fig.set_dpi(300)

    pearson_corr, _ = pearsonr(x, y)
    spearman_corr, _ = spearmanr(x, y)

    plt.subplots_adjust(top=0.9)
    plt.xlabel("Significant caQTL Betas")
    plt.ylabel(ylabel)
    g.figure.suptitle(f'{title}\nPearson: {pearson_corr:.4f} --- Spearman: {spearman_corr:.4f}', 
                x=0.5, y=0.98, ha='center')
    plt.grid()
    plt.show()

    return pearson_corr, spearman_corr  

def effect_size_logfc(filtered_df, title, ylabel="LogFC Scores", yaxis="llm_logfc"):
    x = filtered_df["meanLog2FC"]
    y = filtered_df[yaxis]

    g = sns.jointplot(x=x, y=y, 
                    kind="scatter")
    
    g.fig.set_dpi(300)

    pearson_corr, _ = pearsonr(x, y)
    spearman_corr, _ = spearmanr(x, y)

    plt.subplots_adjust(top=0.9) 
    plt.xlabel("Significant Allele-Specific Binding LogFC")
    plt.ylabel(ylabel)
    g.figure.suptitle(f'{title}\nPearson: {pearson_corr:.4f} --- Spearman: {spearman_corr:.4f}', 
                x=0.5, y=0.98, ha='center')
    plt.grid()
    plt.show()
    return pearson_corr, spearman_corr  

def est_size_logfc(filtered_df, title, ylabel="LogFC Scores", yaxis="llm_logfc"):
    x = filtered_df["obs.estimate"]
    y = filtered_df[yaxis]

    g = sns.jointplot(x=x, y=y, 
                    kind="scatter")
    
    g.fig.set_dpi(300)

    pearson_corr, _ = pearsonr(x, y)
    spearman_corr, _ = spearmanr(x, y)

    plt.subplots_adjust(top=0.9) 
    plt.xlabel("Significant Allele-Specific Binding LogFC")
    plt.ylabel("LogFC Scores")
    g.figure.suptitle(f'{title}\nPearson: {pearson_corr:.4f} --- Spearman: {spearman_corr:.4f}', 
                x=0.5, y=0.98, ha='center')
    plt.grid()
    plt.show()
    return pearson_corr, spearman_corr