import os
import sys
from sklearn.cluster import *
from sklearn.metrics import *
from umap import UMAP
import numpy as np

from .....embedding_clustering import EmbeddingCluster, load_embeddings_and_labels

embedding_file = sys.argv[1]
label_file = sys.argv[2]
out_dir = sys.argv[3]

os.makedirs(out_dir, exist_ok=True)

cluster_metric = adjusted_rand_score

print("Loading embeddings and labels")
embeddings, labels, categories = load_embeddings_and_labels(embedding_file, label_file)
print(embeddings.shape)

cluster_obj = KMeans(n_clusters=len(np.unique(labels)))

print("Performing clustering")
emb_cluster = EmbeddingCluster(cluster_obj, embeddings, labels)

print(emb_cluster.get_clustering_score(cluster_metric))

print("Visualizing")
emb_cluster.plot_embeddings(UMAP(), f"{out_dir}cluster_plot.png", categories)

emb_cluster.save_model(f"{out_dir}cluster_obj.joblib")

