import glob
import numpy as np
import h5py
from scipy import spatial
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import manhattan_distances
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

def load_embeddings_and_compute_cosine_distance(embedding_dir, h5_file, progress_bar=False):
    '''
    Assumes embedding_dir contains ONLY embedding numpy arrays
    Elements in each array will have the same label
    '''
    file = h5py.File(os.path.join(embedding_dir, h5_file))
    cosine_distances = []
    allele_keys = list(file['allele1'].keys())
    if "idx_var" in allele_keys:
        allele_keys.remove('idx_var')
    else:
        allele_keys.remove("idx_fix")
    sorted_list = sorted(allele_keys, key=lambda x: int(x.split('_')[1])) # ensure that they are in the correct order
    for key in tqdm(sorted_list, disable = (not progress_bar)):
        split = key.split("_")
        ind_start, ind_end = int(split[-2]), int(split[-1])
        allele1_array = file['allele1'][key][:]
        allele2_array = file['allele2'][key][:]
        if "idx_var" in list(file['allele1'].keys()):
            idx_vars = file['allele1']['idx_var'][ind_start:ind_end]
            mins1, maxes1 = idx_vars.min(1), idx_vars.max(1) + 1
            indices1 = [np.arange(mi, ma) for mi, ma in zip(mins1, maxes1)]

            idx_vars = file['allele2']['idx_var'][ind_start:ind_end]
            mins2, maxes2 = idx_vars.min(1), idx_vars.max(1) + 1
            indices2 = [np.arange(mi, ma) for mi, ma in zip(mins2, maxes2)]
            
            for i in range(allele1_array.shape[0]):
                a = np.mean(allele1_array[i, indices1[i], :], axis=0) # average over the tokens for each variant (result = batch_size x embedding_size)
                b = np.mean(allele2_array[i, indices2[i], :], axis=0) 
                cosine_distances+=[spatial.distance.cosine(a, b)]

        else:
            idx_fix = file['allele1']['idx_fix']
            min_idx, max_idx = min(idx_fix), max(idx_fix)+1
            for i in range(allele1_array.shape[0]):
                a = np.mean(allele1_array[i, min_idx:max_idx, :], axis=0) # average over the tokens for each variant (result = batch_size x embedding_size)
                b = np.mean(allele2_array[i, min_idx:max_idx, :], axis=0) 
                cosine_distances+=[spatial.distance.cosine(a, b)]

    return cosine_distances

def load_embeddings_and_compute_l1_distance(embedding_dir, h5_file, progress_bar=False):
    '''
    Assumes embedding_dir contains ONLY embedding numpy arrays
    Elements in each array will have the same label
    '''
    file = h5py.File(os.path.join(embedding_dir, h5_file))
    l1_distances = []
    allele_keys = list(file['allele1'].keys())
    if "idx_var" in allele_keys:
        allele_keys.remove('idx_var')
    else:
        allele_keys.remove("idx_fix")
    sorted_list = sorted(allele_keys, key=lambda x: int(x.split('_')[1])) # ensure that they are in the correct order
    for key in tqdm(sorted_list, disable = (not progress_bar)):
        split = key.split("_")
        ind_start, ind_end = int(split[-2]), int(split[-1])
        allele1_array = file['allele1'][key][:]
        allele2_array = file['allele2'][key][:]
        if "idx_var" in list(file['allele1'].keys()):
            idx_vars = file['allele1']['idx_var'][ind_start:ind_end]
            mins1, maxes1 = idx_vars.min(1), idx_vars.max(1) + 1
            indices1 = [np.arange(mi, ma) for mi, ma in zip(mins1, maxes1)]

            idx_vars = file['allele2']['idx_var'][ind_start:ind_end]
            mins2, maxes2 = idx_vars.min(1), idx_vars.max(1) + 1
            indices2 = [np.arange(mi, ma) for mi, ma in zip(mins2, maxes2)]
            
            for i in range(allele1_array.shape[0]):
                a = np.mean(allele1_array[i, indices1[i], :], axis=0) # average over the tokens for each variant (result = batch_size x embedding_size)
                b = np.mean(allele2_array[i, indices2[i], :], axis=0) 
                l1_distances+=[spatial.distance.cosine(a, b)]

        else:
            idx_fix = file['allele1']['idx_fix']
            min_idx, max_idx = min(idx_fix), max(idx_fix)+1
            for i in range(allele1_array.shape[0]):
                a = np.mean(allele1_array[i, min_idx:max_idx, :], axis=0) # average over the tokens for each variant (result = batch_size x embedding_size)
                b = np.mean(allele2_array[i, min_idx:max_idx, :], axis=0) 
                l1_distances+=[manhattan_distances(a, b)]

    return l1_distances

def plot_auprc_auroc_cosine_distances():
    scratch_dir = '/scratch/groups/akundaje/dnalm_benchmark/embeddings/variant_embeddings/'
    model_names = ["Nucleotide-Transformer", "Mistral-DNA", "GenaLM", "HyenaDNA"]# , "DNABERT2"]
    for model in model_names:
        print(model)
        path = glob.glob(os.path.join(scratch_dir, model, '*Afr.CaQTLs*cosine*.tsv'))[0]
        Afr_caQTLs_df = pd.read_csv(path, sep="\t")
        filtered_var_Afr_caQTLs_df = Afr_caQTLs_df[(Afr_caQTLs_df["IsUsed"]==True) & (np.log10(Afr_caQTLs_df["pval"])<3)]
        print(np.unique(filtered_var_Afr_caQTLs_df["label"]))
        filtered_var_Afr_caQTLs_df_true = filtered_var_Afr_caQTLs_df[filtered_var_Afr_caQTLs_df["label"]==1]
        filtered_var_Afr_caQTLs_df_false = filtered_var_Afr_caQTLs_df[filtered_var_Afr_caQTLs_df["label"]==0]
        print(filtered_var_Afr_caQTLs_df_true.shape)
        print(filtered_var_Afr_caQTLs_df_false.shape)
        print("Average Precision: ", average_precision_score(filtered_var_Afr_caQTLs_df["label"], 
                                    filtered_var_Afr_caQTLs_df["cosine_distances"]))
        
        print("AUROC: ", roc_auc_score(filtered_var_Afr_caQTLs_df["label"], 
                                    filtered_var_Afr_caQTLs_df["cosine_distances"]))
        precision, recall, thresholds = precision_recall_curve(filtered_var_Afr_caQTLs_df["label"], 
                                                            filtered_var_Afr_caQTLs_df["cosine_distances"])
        
        FPR, TPR, _ = roc_curve(filtered_var_Afr_caQTLs_df["label"], filtered_var_Afr_caQTLs_df["cosine_distances"], pos_label=1)
        
        plt.plot(FPR, TPR)
        plt.show()
        
        auprc = auc(recall, precision)
        print("AUPRC:", auprc)
        plt.plot(recall, precision)
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.title("Afr CaQTLs: Precision-Recall curve");
        plt.show()
        
        plt.hist([filtered_var_Afr_caQTLs_df_true["cosine_distances"],
                filtered_var_Afr_caQTLs_df_false["cosine_distances"]], 
                color=['Red', 'Blue'], label=['Significant', 'Control'])
        plt.legend()
        plt.show()

def sig_ctrl_variants_Eu_CaQTLs(likelihoods_data_path):
    eu_caqtls_data_path =  "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/Eu.CaQTLS.tsv"
    eu_caQTLs_df = pd.read_csv(eu_caqtls_data_path, sep="\t")
    likelihood = pd.read_csv(likelihoods_data_path, sep="\t")
    threshold = 3
    if eu_caQTLs_df.shape[0] == likelihood.shape[0]:
        likelihoods_data = pd.concat([eu_caQTLs_df, likelihood], axis=1)
        filtered_var_eu_caQTLs_df = likelihoods_data[(likelihoods_data["Inside_Peak"]==True) &
                                            (likelihoods_data["IsUsed"]==True)].copy(deep=True)
        
        filtered_var_eu_caQTLs_df["llm_logfc"] = np.log(filtered_var_eu_caQTLs_df["allele1_likelihoods"]/filtered_var_eu_caQTLs_df["allele2_likelihoods"])

        filtered_var_eucaqtls_df_ctrl = filtered_var_eu_caQTLs_df[filtered_var_eu_caQTLs_df["Log10_BF"]<-1].copy(deep=True)
        filtered_var_eucaqtls_df_sig = filtered_var_eu_caQTLs_df[filtered_var_eu_caQTLs_df["Log10_BF"]>threshold].copy(deep=True)

        ctrl_likelihoods = np.abs(filtered_var_eucaqtls_df_ctrl["llm_logfc"]) # np.abs(np.log(filtered_var_eucaqtls_df_ctrl["allele1_likelihoods"]/filtered_var_eucaqtls_df_ctrl["allele2_likelihoods"]))
        sig_likelihoods = np.abs(filtered_var_eucaqtls_df_sig["llm_logfc"])# np.abs(np.log(filtered_var_eucaqtls_df_sig["allele1_likelihoods"]/filtered_var_eucaqtls_df_sig["allele2_likelihoods"]))

        print(len(ctrl_likelihoods), len(sig_likelihoods))

        counts_ctrl, bins_ctrl = np.histogram(ctrl_likelihoods, bins=100)
        # fractions_ctrol = counts_ctrl / counts_ctrl.sum()
        # plt.hist(bins_ctrl[:-1], bins_ctrl, weights=fractions_ctrol, alpha=0.7, label="control")

        counts_sig, bins_sig = np.histogram(sig_likelihoods, bins=100)
        # fractions_sig = counts_sig / counts_sig.sum()
        # plt.hist(bins_sig[:-1], bins_sig, weights=fractions_sig, alpha=0.7, label="significant")

        # U1, p = mannwhitneyu(counts_ctrl, counts_sig, alternative="greater")

        return ctrl_likelihoods, sig_likelihoods, filtered_var_eu_caQTLs_df
    
def sig_ctrl_variants_Eu_CaQTLs_probed_counts(counts_data_path):
    threshold = 3
    
    counts_data = pd.read_csv(counts_data_path, sep="\t")
    filtered_var_eu_caQTLs_df = counts_data[(counts_data["Inside_Peak"]==True) &
                                        (counts_data["IsUsed"]==True)].copy(deep=True)
    
    filtered_var_eu_caQTLs_df["llm_logfc"] = filtered_var_eu_caQTLs_df["allele1_counts"] - filtered_var_eu_caQTLs_df["allele2_counts"]

    filtered_var_eucaqtls_df_ctrl = filtered_var_eu_caQTLs_df[filtered_var_eu_caQTLs_df["Log10_BF"]<-1].copy(deep=True)
    filtered_var_eucaqtls_df_sig = filtered_var_eu_caQTLs_df[filtered_var_eu_caQTLs_df["Log10_BF"]>threshold].copy(deep=True)

    ctrl_likelihoods = np.abs(filtered_var_eucaqtls_df_ctrl["llm_logfc"]) # np.abs(np.log(filtered_var_eucaqtls_df_ctrl["allele1_likelihoods"]/filtered_var_eucaqtls_df_ctrl["allele2_likelihoods"]))
    sig_likelihoods = np.abs(filtered_var_eucaqtls_df_sig["llm_logfc"])# np.abs(np.log(filtered_var_eucaqtls_df_sig["allele1_likelihoods"]/filtered_var_eucaqtls_df_sig["allele2_likelihoods"]))

    print(len(ctrl_likelihoods), len(sig_likelihoods))

    counts_ctrl, bins_ctrl = np.histogram(ctrl_likelihoods, bins=100)
    # fractions_ctrol = counts_ctrl / counts_ctrl.sum()
    # plt.hist(bins_ctrl[:-1], bins_ctrl, weights=fractions_ctrol, alpha=0.7, label="control")

    counts_sig, bins_sig = np.histogram(sig_likelihoods, bins=100)
    # fractions_sig = counts_sig / counts_sig.sum()
    # plt.hist(bins_sig[:-1], bins_sig, weights=fractions_sig, alpha=0.7, label="significant")

    # U1, p = mannwhitneyu(counts_ctrl, counts_sig, alternative="greater")

    return ctrl_likelihoods, sig_likelihoods, filtered_var_eu_caQTLs_df

def sig_ctrl_variants_Afr_CaQTLs(likelihood_data_path):
    afr_caqtls_data_path =  "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/Afr.CaQTLS.tsv"
    afr_caQTLs_df = pd.read_csv(afr_caqtls_data_path, sep="\t")
    likelihood = pd.read_csv(likelihood_data_path, sep="\t")
    if afr_caQTLs_df.shape[0] == likelihood.shape[0]:
        likelihoods_data = pd.concat([afr_caQTLs_df, likelihood], axis=1)
        filtered_var_afr_caQTLs_df = likelihoods_data[(likelihoods_data["IsUsed"]==True) & (np.log10(likelihoods_data["pval"])<3)].copy(deep=True)
        filtered_var_afr_caQTLs_df["llm_logfc"] = np.log(filtered_var_afr_caQTLs_df["allele1_likelihoods"]/filtered_var_afr_caQTLs_df["allele2_likelihoods"])

        print("unique label values", np.unique(filtered_var_afr_caQTLs_df["label"]))
        filtered_var_afrcaqtls_df_sig = filtered_var_afr_caQTLs_df[filtered_var_afr_caQTLs_df["label"]==1]
        filtered_var_afrcaqtls_df_ctrl = filtered_var_afr_caQTLs_df[filtered_var_afr_caQTLs_df["label"]==0]

        control_likelihoods = np.abs(filtered_var_afrcaqtls_df_ctrl["llm_logfc"]) # np.abs(np.log(filtered_var_afrcaqtls_df_ctrl["allele1_likelihoods"]/filtered_var_afrcaqtls_df_ctrl["allele2_likelihoods"]))
        sig_likelihoods = np.abs(filtered_var_afrcaqtls_df_sig["llm_logfc"])  # np.abs(np.log(filtered_var_afrcaqtls_df_sig["allele1_likelihoods"]/filtered_var_afrcaqtls_df_sig["allele2_likelihoods"]))

        print(len(control_likelihoods), len(sig_likelihoods))

        counts_ctrl, bins_ctrl = np.histogram(control_likelihoods, bins=100)
        # fractions_ctrol = counts_ctrl / counts_ctrl.sum()
        # plt.hist(bins_ctrl[:-1], bins_ctrl, weights=fractions_ctrol, alpha=0.7, label="control")

        counts_sig, bins_sig = np.histogram(sig_likelihoods, bins=100)
        # fractions_sig = counts_sig / counts_sig.sum()
        # plt.hist(bins_sig[:-1], bins_sig, weights=fractions_sig, alpha=0.7, label="significant")

        U1, p = mannwhitneyu(counts_ctrl, counts_sig, alternative="greater")

        return control_likelihoods, sig_likelihoods, filtered_var_afr_caQTLs_df
    
def sig_ctrl_variants_Afr_CaQTLs_probed_counts(counts_data_path):
    
    counts_data = pd.read_csv(counts_data_path, sep="\t")
    filtered_var_afr_caQTLs_df = counts_data[(counts_data["IsUsed"]==True) & (np.log10(counts_data["pval"])<3)].copy(deep=True)
    filtered_var_afr_caQTLs_df["llm_logfc"] = filtered_var_afr_caQTLs_df["allele1_counts"]-filtered_var_afr_caQTLs_df["allele2_counts"]

    print("unique label values", np.unique(filtered_var_afr_caQTLs_df["label"]))
    filtered_var_afrcaqtls_df_sig = filtered_var_afr_caQTLs_df[filtered_var_afr_caQTLs_df["label"]==1]
    filtered_var_afrcaqtls_df_ctrl = filtered_var_afr_caQTLs_df[filtered_var_afr_caQTLs_df["label"]==0]

    control_counts = np.abs(filtered_var_afrcaqtls_df_ctrl["llm_logfc"]) # np.abs(np.log(filtered_var_afrcaqtls_df_ctrl["allele1_likelihoods"]/filtered_var_afrcaqtls_df_ctrl["allele2_likelihoods"]))
    sig_counts = np.abs(filtered_var_afrcaqtls_df_sig["llm_logfc"])  # np.abs(np.log(filtered_var_afrcaqtls_df_sig["allele1_likelihoods"]/filtered_var_afrcaqtls_df_sig["allele2_likelihoods"]))

    print(len(control_counts), len(sig_counts))

    counts_ctrl, bins_ctrl = np.histogram(control_counts, bins=100)
    # fractions_ctrol = counts_ctrl / counts_ctrl.sum()
    # plt.hist(bins_ctrl[:-1], bins_ctrl, weights=fractions_ctrol, alpha=0.7, label="control")

    counts_sig, bins_sig = np.histogram(sig_counts, bins=100)
    # fractions_sig = counts_sig / counts_sig.sum()
    # plt.hist(bins_sig[:-1], bins_sig, weights=fractions_sig, alpha=0.7, label="significant")

    U1, p = mannwhitneyu(counts_ctrl, counts_sig, alternative="greater")

    return control_counts, sig_counts, filtered_var_afr_caQTLs_df
    
def variants_Afr_ASB_CaQTLs(likelihood_data_path):
    afr_caqtls_data_path =  "/oak/stanford/groups/akundaje/anusri/variant-benchmakring/Afr.ASB.CaQTLS.tsv"
    afr_caQTLs_df = pd.read_csv(afr_caqtls_data_path, sep="\t")
    likelihood = pd.read_csv(likelihood_data_path, sep="\t")
    if afr_caQTLs_df.shape[0] == likelihood.shape[0]:
        likelihoods_data = pd.concat([afr_caQTLs_df, likelihood], axis=1)
        # filtered_var_afr_caQTLs_df = likelihoods_data[likelihoods_data["IsUsed"]==True].copy(deep=True)
        filtered_var_afr_caQTLs_df = likelihoods_data.copy(deep=True)
        filtered_var_afr_caQTLs_df["llm_logfc"] = np.log(filtered_var_afr_caQTLs_df["allele1_likelihoods"]/filtered_var_afr_caQTLs_df["allele2_likelihoods"])

    return filtered_var_afr_caQTLs_df

def variants_Afr_ASB_CaQTLs_probed_counts(counts_data_path):
    counts_data = pd.read_csv(counts_data_path, sep="\t")
    filtered_var_afr_caQTLs_df = counts_data.copy(deep=True)
    print(filtered_var_afr_caQTLs_df.columns)
    filtered_var_afr_caQTLs_df["llm_logfc"] = filtered_var_afr_caQTLs_df["allele1_counts"] - filtered_var_afr_caQTLs_df["allele2_counts"]

    return filtered_var_afr_caQTLs_df

def beta_logfc(filtered_df, title, ylabel="LogFC Scores"):
    if "Beta" in filtered_df.columns:
        x = filtered_df["Beta"]
    else:
        x = filtered_df["beta"]
    y = filtered_df["llm_logfc"]
    g = sns.jointplot(x=x, y=y, 
                    kind="scatter")

    pearson_corr, _ = pearsonr(x, y)
    spearman_corr, _ = spearmanr(x, y)

    # Add the correlation coefficients to the plot
    plt.subplots_adjust(top=0.9)  # Adjust the top edge of the subplot to make room for the text
    plt.xlabel("Significant caQTL Betas")
    plt.ylabel(ylabel)
    g.figure.suptitle(f'{title}\nPearson: {pearson_corr:.4f} --- Spearman: {spearman_corr:.4f}', 
                x=0.5, y=0.98, ha='center')
    plt.grid()
    plt.show()

def effect_size_logfc(filtered_df, title):
    x = filtered_df["meanLog2FC"]
    y = filtered_df["llm_logfc"]
    g = sns.jointplot(x=x, y=y, 
                    kind="scatter")

    pearson_corr, _ = pearsonr(x, y)
    spearman_corr, _ = spearmanr(x, y)

    # Add the correlation coefficients to the plot
    plt.subplots_adjust(top=0.9)  # Adjust the top edge of the subplot to make room for the text
    plt.xlabel("Significant Allele-Specific Binding LogFC")
    plt.ylabel("LogFC Scores")
    g.figure.suptitle(f'{title}\nPearson: {pearson_corr:.4f} --- Spearman: {spearman_corr:.4f}', 
                x=0.5, y=0.98, ha='center')
    plt.grid()
    plt.show()