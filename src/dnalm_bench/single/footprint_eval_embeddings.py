import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import wilcoxon
import argparse
import os 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def parse_args():
	parser = argparse.ArgumentParser(description="Given a list of motif-inserted sequences and their embeddings, determine model performance on each motif")
	parser.add_argument("--input_seqs", type=str, required=True, help="Text file containing input negative elements which were footprinted")
	parser.add_argument("--embeddings", type=str, required=True, help="h5 file containing model embeddings")
	parser.add_argument("--output_file", type=str, required=True, help="Output file path containing metrics")
	args = parser.parse_args()
	return args



def load_embeddings(emb_h5):
	running_arrays = []
	for key in list(emb_h5['seq'].keys()):
	    if "idx" in key:
	        continue
	    split = key.split("_")
	    ind_start, ind_end = int(split[-2]), int(split[-1])
	    h5_array = emb_h5['seq'][key][:]
	    if "idx_var" in emb_h5['seq'].keys():
	        idx_vars = emb_h5['seq']['idx_var'][ind_start:ind_end]
	        mins, maxes = idx_vars.min(1), idx_vars.max(1) + 1
	        indices = [np.arange(mi, ma) for mi, ma in zip(mins, maxes)]
	        curr_means = np.array([np.mean(h5_array[i, indices[i], :], axis=0) for i in range(h5_array.shape[0])])
	        running_arrays.append(curr_means)
	    elif "idx_fix" in emb_h5['seq'].keys():
	        idx_fix = emb_h5['seq']['idx_fix'][:]
	        indices = np.arange(idx_fix.min(), idx_fix.max() + 1)
	        curr_means = np.mean(h5_array[:, indices, :], axis=1)
	        running_arrays.append(curr_means)

	embedding_array = np.vstack(running_arrays)
	return embedding_array


def relate_embeddings_to_motifs(embedding_array, seq_data):
	embeddings_dict = {}
	for seq_ind in range(len(embedding_array)):
	    seq_key = seq_data.loc[seq_ind, 0].split("_")
	    if seq_key[0] not in embeddings_dict:
	        embeddings_dict[seq_key[0]] = {}
	    curr_subdict = embeddings_dict[seq_key[0]].get(seq_key[1], {})
	    if seq_key[1] == "raw":
	        curr_subdict[seq_key[2]] = embedding_array[seq_ind]
	        embeddings_dict[seq_key[0]][seq_key[1]] = curr_subdict
	        continue
	    if len(curr_subdict) == 0:
	        curr_subdict["forward"] = {}
	        curr_subdict["reverse"] = {}
	    curr_subdict[seq_key[2]][seq_key[3]] = embedding_array[seq_ind]
	    embeddings_dict[seq_key[0]][seq_key[1]] = curr_subdict
	return embeddings_dict


def get_distances(embeddings_dict):
	distances_dict = {x:{"true":[], "shuffled":[]} for x in embeddings_dict["0"].keys() if x != "raw"}
	for seq in embeddings_dict:
	    seq_dict = embeddings_dict[seq]
	    for motif in seq_dict:
	        if motif == "raw":
	            continue
	        distances_dict[motif]["true"].append(cosine(seq_dict["raw"]["forward"], seq_dict[motif]["forward"]["true"]))
	        distances_dict[motif]["true"].append(cosine(seq_dict["raw"]["reverse"], seq_dict[motif]["reverse"]["true"]))
	        distances_dict[motif]["shuffled"].append(cosine(seq_dict["raw"]["forward"], seq_dict[motif]["forward"]["shuffled"]))
	        distances_dict[motif]["shuffled"].append(cosine(seq_dict["raw"]["reverse"], seq_dict[motif]["reverse"]["shuffled"]))
	return distances_dict


def get_accuracies(distances_dict):
	accuracy_dict = {}
	for seq in distances_dict:
	    true_distances = distances_dict[seq]["true"]
	    shuffled_distances = distances_dict[seq]["shuffled"]
	    accuracy_dict[seq] = sum([true_distances[x] > shuffled_distances[x] for x in range(len(true_distances))]) / len(true_distances)
	return accuracy_dict

def get_pvals(distances_dict):
	pval_dict = {}
	for seq in distances_dict:
	    true_distances = distances_dict[seq]["true"]
	    shuffled_distances = distances_dict[seq]["shuffled"]
	    pval_dict[seq] = wilcoxon(true_distances, shuffled_distances, alternative="greater")[1]
	return pval_dict

def main():
	args = parse_args()
	seq_data = pd.read_csv(args.input_seqs, sep="\t", header=None)
	emb_h5 = h5py.File(args.embeddings, "r")
	embedding_array = load_embeddings(emb_h5)
	motif_embedding_dict = relate_embeddings_to_motifs(embedding_array, seq_data)
	distances_dict = get_distances(motif_embedding_dict)
	accuracies, pvals = get_accuracies(distances_dict), get_pvals(distances_dict)
	final_df = pd.DataFrame([accuracies, pvals], index=["Accuracy", "P-Value"]).T
	final_df.to_csv(args.output_file, sep="\t", index=True, header=True)

if __name__ == "__main__":
	main()
    

        

