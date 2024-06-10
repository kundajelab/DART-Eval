import tensorflow as tf
import tensorflow_probability as tfp
import pyBigWig
import pyfaidx
from tensorflow import keras
import math
import json
import pandas as pd 
import numpy as np 
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score


def mean_squared_error(true_vals, pred_vals):
	return np.mean(np.square(true_vals - pred_vals))


def dna_to_one_hot(seqs):
	seq_len = len(seqs[0])
	assert np.all(np.array([len(s) for s in seqs]) == seq_len)

	# Join all sequences together into one long string, all uppercase
	seq_concat = "".join(seqs).upper() + "ACGT"
	# Add one example of each base, so np.unique doesn't miss indices later

	one_hot_map = np.identity(5)[:, :-1].astype(np.int8)

	# Convert string into array of ASCII character codes;
	base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)

	# Anything that's not an A, C, G, or T gets assigned a higher code
	base_vals[~np.isin(base_vals, np.array([65, 67, 71, 84]))] = 85

	# Convert the codes into indices in [0, 4], in ascending order by code
	_, base_inds = np.unique(base_vals, return_inverse=True)

	# Get the one-hot encoding for those indices, and reshape back to separate
	return one_hot_map[base_inds[:-4]].reshape((len(seqs), seq_len, 4))

def multinomial_nll(true_counts, logits):
	"""Compute the multinomial negative log-likelihood
	Args:
	  true_counts: observed count values
	  logits: predicted logit values
	"""
	counts_per_example = tf.reduce_sum(true_counts, axis=-1)
	dist = tfp.distributions.Multinomial(total_count=counts_per_example,
										 logits=logits)
	return (-tf.reduce_sum(dist.log_prob(true_counts)) / 
			tf.cast(tf.shape(true_counts)[0], dtype=tf.float32))


def load_chrombpnet_model(model_file):
	with keras.utils.CustomObjectScope({'multinomial_nll':multinomial_nll, 'tf':tf}):
		model = keras.models.load_model(model_file)
	return model 


class ChromBPNetPeakDataset(keras.utils.Sequence):
	def __init__(self, peak_file, bigwig, genome_fa, chroms, batch_size):
		self.chroms = chroms 
		self.peak_table = self.load_peak_table(peak_file)
		self.bigwig = pyBigWig.open(bigwig)
		self.genome = pyfaidx.Fasta(genome_fa, sequence_always_upper=True)
		self.batch_size = batch_size
		self.seqs, self.values = self.load_seqs_and_signal()

	def load_peak_table(self, peak_file):
		peak_table = pd.read_csv(peak_file, sep="\t")
		if self.chroms is not None:
			peak_table = peak_table.loc[peak_table["chr"].isin(self.chroms)].reset_index()
		return peak_table

	def load_seqs_and_signal(self):
		dna_seqs, bigwig_data = [], []
		for peak_idx in range(len(self.peak_table)):
			chrom, start, end, out_start, out_end = self.peak_table.loc[peak_idx, "chr"], self.peak_table.loc[peak_idx, "input_start"], self.peak_table.loc[peak_idx, "input_end"], self.peak_table.loc[peak_idx, "elem_start"], self.peak_table.loc[peak_idx, "elem_end"]
			dna_seqs.append(self.genome[chrom][start:end].seq)
			bigwig_data.append(np.nan_to_num(self.bigwig.values(chrom, int(out_start), int(out_end))))

		seqs_array, bigwig_array = dna_to_one_hot(dna_seqs), np.array(bigwig_data)
		assert list(seqs_array.shape) == [len(self.peak_table), 2114, 4]
		assert list(bigwig_array.shape) == [len(self.peak_table), 1000]

		return seqs_array, bigwig_array

	def __len__(self):
		return math.ceil(len(self.peak_table) / self.batch_size)

	def __getitem__(self, idx):
		idx_start = self.batch_size * idx 
		idx_end = min(len(self.peak_table), self.batch_size*(idx+1))

		return self.seqs[idx_start:idx_end], np.log(1+self.values[idx_start:idx_end].sum(-1, keepdims=True))



def get_counts_predictions(model, peak_dataset):
	true_counts, pred_counts = [], []
	num_batches = len(peak_dataset)
	for idx in range(num_batches):
		if idx%100==0:
			print(str(idx)+'/'+str(num_batches))        
		seqs, counts = peak_dataset[idx]

		preds = model.predict_on_batch(seqs)
		true_counts.extend(counts)
		pred_counts.extend(preds[1][:,0])
	print(len(true_counts), len(pred_counts))
	return np.array(true_counts).flatten(), np.array(pred_counts)


def calc_metrics(pos_true, pos_pred, neg_true, neg_pred, idr_true, idr_pred, out_path):
	all_true = np.concatenate([pos_true, neg_true])
	all_pred = np.concatenate([pos_pred, neg_pred])

	test_loss_pos = mean_squared_error(pos_true, pos_pred)
	test_loss_neg = mean_squared_error(neg_true, neg_pred)
	test_loss_idr = mean_squared_error(idr_true, idr_pred)
	test_loss_all = mean_squared_error(all_true, all_pred)

	test_pearson_pos = pearsonr(pos_true, pos_pred)[0]
	test_pearson_neg = pearsonr(neg_true, neg_pred)[0]
	test_pearson_idr = pearsonr(idr_true, idr_pred)[0]
	test_pearson_all = pearsonr(all_true, all_pred)[0]

	test_spearman_pos = spearmanr(pos_true, pos_pred)[0]
	test_spearman_neg = spearmanr(neg_true, neg_pred)[0]
	test_spearman_idr = spearmanr(idr_true, idr_pred)[0]
	test_spearman_all = spearmanr(all_true, all_pred)[0]

	test_labels = np.concatenate([np.ones_like(idr_true), np.zeros_like(neg_true)])
	test_counts_pred_cls = np.concatenate([idr_pred, neg_pred])
	test_auroc = roc_auc_score(test_labels, test_counts_pred_cls)
	test_auprc = average_precision_score(test_labels, test_counts_pred_cls)

	metrics = {
	"test_loss_pos": test_loss_pos,
	"test_loss_idr": test_loss_idr,
	"test_loss_neg": test_loss_neg,
	"test_loss_all": test_loss_all,
	"test_pearson_pos": test_pearson_pos,
	"test_pearson_idr": test_pearson_idr,
	"test_pearson_neg": test_pearson_neg,
	"test_pearson_all": test_pearson_all,
	"test_spearman_pos": test_spearman_pos,
	"test_spearman_idr": test_spearman_idr,
	"test_spearman_neg": test_spearman_neg,
	"test_spearman_all": test_spearman_all,
	"test_auroc": test_auroc,
	"test_auprc": test_auprc,
	}

	print(metrics)
	with open(out_path, "w") as f:
		json.dump(metrics, f, indent=4)

	return metrics






