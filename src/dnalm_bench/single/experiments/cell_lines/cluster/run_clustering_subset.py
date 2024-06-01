import os
import sys
from sklearn.cluster import *
from sklearn.metrics import *
from sklearn.decomposition import *
from umap import UMAP
import numpy as np

np.random.seed(0)
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
# cluster_obj = KMeans(n_clusters=n_clusters)

print("Performing clustering")

cluster_objs = [EmbeddingCluster(KMeans(n_clusters=n_clusters, random_state=it), embeddings, labels) for it in range(100)]
scores = [emb_cluster.get_clustering_score(cluster_metric) for emb_cluster in cluster_objs]

scores_mean = np.mean(scores)
scores_cint = np.max([np.abs(scores_mean - np.quantile(scores, 0.025)), np.abs(scores_mean - np.quantile(scores, 0.975))])
print(scores_mean, scores_cint)
# print(np.mean(scores), 1.96 * np.std(scores))


# emb_cluster = EmbeddingCluster(cluster_obj, embeddings, labels)

# print(emb_cluster.get_clustering_score(cluster_metric))

# print("Visualizing")
cluster_objs[0].plot_embeddings(UMAP(), f"{out_dir}cluster_plot.png", categories)

# emb_cluster.save_model(f"{out_dir}cluster_obj.joblib")

