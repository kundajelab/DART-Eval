import os
import sys
from sklearn.cluster import *
from sklearn.metrics import *
from sklearn.decomposition import *
from umap import UMAP
import numpy as np

from .....embedding_clustering import EmbeddingCluster, load_embeddings_and_labels, load_embeddings_and_labels_subset

embedding_file = sys.argv[1]
label_file = sys.argv[2]
index_file = sys.argv[3]
out_dir = sys.argv[4]

os.makedirs(out_dir, exist_ok=True)

cluster_metric = adjusted_mutual_info_score

print("Loading embeddings and labels")
embeddings, labels, categories = load_embeddings_and_labels_subset(embedding_file, label_file, index_file)
print(embeddings.shape)

embeddings = PCA(n_components=60).fit_transform(embeddings)
n_clusters = 50
print(n_clusters)
cluster_obj = KMeans(n_clusters=n_clusters)

print("Performing clustering")
emb_cluster = EmbeddingCluster(cluster_obj, embeddings, labels)

print(emb_cluster.get_clustering_score(cluster_metric))

print("Visualizing")
emb_cluster.plot_embeddings(UMAP(), f"{out_dir}cluster_plot.png", categories)

emb_cluster.save_model(f"{out_dir}cluster_obj.joblib")

