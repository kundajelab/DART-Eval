import os
import sys
from sklearn.cluster import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sknetwork.clustering import Louvain
from sklearn.feature_extraction.text import TfidfTransformer
from umap import UMAP
import numpy as np
import pyfaidx
import pandas as pd
import torch
from .....embedding_clustering import EmbeddingCluster, load_embeddings_and_labels

out_dir = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/clusters_differential/motif_baseline/"

count_table = pd.read_csv("/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/motif_hits/motif_count_matrix_total_hits.tsv", sep="\t", index_col=0)
# count_table = count_table.filter(regex="SPI1|HNF4|GATA|AP1")

peak_file = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/peaks_by_cell_label_unique_dataloader_format.tsv"
peak_table = pd.read_csv(peak_file, sep="\t")
cat_list = list(peak_table["label"].values)
categories = list(set(cat_list))
labels = [categories.index(x) for x in cat_list]

embeddings = count_table.values
# embeddings_tf = TfidfTransformer().fit_transform(embeddings)
# embeddings = embeddings_tf.toarray()
print("Converted to array")
embeddings = PCA().fit_transform(embeddings)[:,:100]

cluster_obj = KMeans(n_clusters=5)

print("Performing clustering")
print(embeddings.shape)
cluster_metric = adjusted_rand_score
emb_cluster = EmbeddingCluster(cluster_obj, embeddings, labels)

print(emb_cluster.get_clustering_score(cluster_metric))

print("Visualizing")
# emb_cluster.plot_embeddings(UMAP(), f"{out_dir}cluster_plot.png", categories)

emb_cluster.save_model(f"{out_dir}cluster_obj.joblib")
