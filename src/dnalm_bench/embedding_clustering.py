from abc import ABCMeta, abstractmethod
import numpy as np
import h5py
import hashlib
from scipy.stats import wilcoxon
import os
import matplotlib.pyplot as plt
from sklearn.cluster import *


class EmbeddingCluster():
	def __init__(self, cluster_obj, embeddings, labels):
		self.cluster_obj = cluster_obj
		self.embeddings = embeddings
		self.true_labels = labels
		self.cluster_obj.fit(self.embeddings)

	def get_cluster_labels(self):
		return self.cluster_obj.labels_

	def get_clustering_score(self, metric):
		cluster_labels = self.get_cluster_labels()
		return metric(self.true_labels, cluster_labels)
	
	def plot_embeddings(self, reduction_obj, out_path, categories):
		embeddings_2d = reduction_obj.fit_transform(self.embeddings)
		first_dim, second_dim = embeddings_2d[:,0], embeddings_2d[:,1]
		plot_labels=[categories[x] for x in self.true_labels]
		plt.figure(dpi=300)
		scatter = plt.scatter(first_dim, second_dim, c=self.true_labels, s=0.5)
		plt.xticks([])
		plt.yticks([])
		plt.title("Model Embeddings Colored by Cell Type")
		print(scatter.legend_elements()[0])
		plt.legend(handles=scatter.legend_elements()[0], labels=categories)
		plt.savefig(out_path, format="svg")
		plt.show()


def load_embeddings_and_labels(embedding_dir, chunk_size=10000):
	'''
	Assumes embedding_dir contains ONLY embedding numpy arrays
	Elements in each array will have the same label
	'''
	labels, arrays = [], []
	cat_list = [x[:-3] for x in os.listdir(embedding_dir) if ".h5" in x]
	print(cat_list)
	emb_files = [h5py.File(os.path.join(embedding_dir, x), "r") for x in os.listdir(embedding_dir) if ".h5" in x]
	for cat, file in enumerate(emb_files):
		running_arrays = []
		num_embs = file["seq_emb"].shape
		# num_embs = [2000, num_embs[1], num_embs[2]]
		curr_index = 0
		while curr_index < num_embs[0]:
			print(curr_index)
			curr_chunk = file["seq_emb"][curr_index : min(curr_index+chunk_size, num_embs[0])]
			running_arrays.append(curr_chunk.mean(1))
			curr_index += chunk_size
		arrays.append(np.vstack(running_arrays))
		labels.extend([cat] * num_embs[0])

	return np.vstack(arrays), labels, cat_list