import os
import sys
from sklearn.cluster import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from umap import UMAP
import numpy as np
import pyfaidx
import pandas as pd
import torch

from .....embedding_clustering import EmbeddingCluster, load_embeddings_and_labels

def kmers(X, k, scores=None):

	n = X.shape[1]

	X = X.type(torch.int32)
	w = torch.arange(n).repeat(k, 1).T * n ** torch.arange(k)
	w = w[None, :, :].type(torch.int32).to(X.device)
	idxs = torch.nn.functional.conv1d(X, w).type(torch.int64)[:, 0]

	if scores is not None:
		scores = scores.unsqueeze(1).type(torch.float32)
		ws = torch.ones(1, 1, k, dtype=torch.float32)
		score_ = torch.nn.functional.conv1d(scores, ws)[:, 0]
	else:
		score_ = torch.ones(1, dtype=torch.float32).expand_as(idxs)

	X_kmers = torch.zeros((X.shape[0], n**k))
	X_kmers.scatter_add_(1, idxs, score_)
	return X_kmers

def dna_to_one_hot(seqs):
    seq_len = len(seqs[0])
    assert np.all(np.array([len(s) for s in seqs]) == seq_len)

    # Join all sequences together into one long string, all uppercase
    seq_concat = "".join(seqs).upper()

    one_hot_map = np.identity(5)[:, :-1]

    # Convert string into array of ASCII character codes;
    base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)

    # Anything that's not an A, C, G, or T gets assigned a higher code
    base_vals[~np.isin(base_vals, np.array([65, 67, 71, 84]))] = 85

    # Convert the codes into indices in [0, 4], in ascending order by code
    _, base_inds = np.unique(base_vals, return_inverse=True)

    # Get the one-hot encoding for those indices, and reshape back to separate
    return one_hot_map[base_inds].reshape((len(seqs), seq_len, 4))


genome_fa = "/oak/stanford/groups/akundaje/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
fasta = pyfaidx.Fasta(genome_fa, sequence_always_upper=True)
peak_file = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/peaks_by_cell_label_unique_dataloader_format.tsv"
peak_table = pd.read_csv(peak_file, sep="\t")
out_dir = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/clusters_differential/kmer_baseline/"

os.makedirs(out_dir, exist_ok=True)

cluster_metric = adjusted_rand_score

print("Loading embeddings and labels")
cat_list = list(peak_table["label"].values)
categories = list(set(cat_list))
labels = [categories.index(x) for x in cat_list]

dna_seqs = []
for peak_ind in range(len(peak_table)):
    dna_seqs.append(fasta[peak_table.loc[peak_ind, "chr"]][peak_table.loc[peak_ind, "input_start"]:peak_table.loc[peak_ind, "input_end"]].seq)

seq_arr = dna_to_one_hot(dna_seqs)
seq_tens = torch.tensor(seq_arr).transpose(1,2)

embeddings = kmers(seq_tens, 6).numpy()
# embeddings = MinMaxScaler().fit_transform(embeddings)
kmer_sums = embeddings.sum(0)
high_indices = kmer_sums.argsort()[:1000]
embeddings = embeddings[:,high_indices]
# embeddings = PCA().fit_transform(embeddings)[:,:16]

cluster_obj = KMeans(n_clusters=len(np.unique(labels)))

print("Performing clustering")
print(embeddings.shape)
emb_cluster = EmbeddingCluster(cluster_obj, embeddings, labels)

print(emb_cluster.get_clustering_score(cluster_metric))

print("Visualizing")
emb_cluster.plot_embeddings(UMAP(), f"{out_dir}cluster_plot.png", categories)

emb_cluster.save_model(f"{out_dir}cluster_obj.joblib")

