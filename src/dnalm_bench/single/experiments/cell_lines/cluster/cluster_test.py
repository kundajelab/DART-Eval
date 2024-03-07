import os
import sys
from sklearn.cluster import *
from sklearn.metrics import *
from umap import UMAP
import numpy as np

from .....embedding_clustering import EmbeddingCluster, load_embeddings_and_labels

embedding_dir = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/embeddings/cell_line_2114/Mistral-DNA-v0.1/"


cluster_metric = adjusted_rand_score

print("Loading embeddings and labels")
embeddings, labels, categories = load_embeddings_and_labels(embedding_dir)
print(embeddings.shape)

cluster_obj = KMeans(n_clusters=len(np.unique(labels)))

print("Performing clustering")
emb_cluster = EmbeddingCluster(cluster_obj, embeddings, labels)

print(emb_cluster.get_clustering_score(cluster_metric))

print("Visualizing")
emb_cluster.plot_embeddings(UMAP(), "/users/patelas/scratch/cluster_test_plot.svg", categories)

