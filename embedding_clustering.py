from abc import ABCMeta, abstractmethod
import numpy as np
import components
import h5py
import hashlib
from scipy.stats import wilcoxon
import os
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
	
	def plot_embeddings(self, reduction_obj, out_path):
		embeddings_2d = reduction_obj.fit_transform(self.embeddings)
		first_dim, second_dim = embeddings_2d[:,0], embeddings_2d[:,1]
		plt.figure(dpi=300)
		plt.scatter(first_dim, second_dim, c=self.true_labels)
		plt.savefig(out_path, format="svg")
		plt.show()


def load_embeddings_and_labels(embedding_dir):
	'''
	Assumes embedding_dir contains ONLY embedding numpy arrays
	Elements in each array will have the same label
	'''
	labels, arrays = [], []
	emb_files = [h5py.File(x, "r") for x in os.listdir(embedding_dir) if ".h5" in x]
	for cat, file in emb_files:
		curr_embs = file["seq_emb"][:]
		arrays.append(curr_embs)
		labels.extend([cat] * len(curr_embs))

	return np.vstack(arrays), labels