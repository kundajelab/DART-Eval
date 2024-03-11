import numpy as np
import h5py
from scipy import spatial
import os
from tqdm import tqdm
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

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