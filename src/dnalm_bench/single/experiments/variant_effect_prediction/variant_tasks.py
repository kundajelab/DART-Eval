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