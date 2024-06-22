import tensorflow as tf
import tensorflow_hub as hub
import pyBigWig
import pyfaidx
from tensorflow import keras
import math
import json
import pandas as pd 
from tqdm import tqdm
import numpy as np 
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score

def mean_squared_error(true_vals, pred_vals):
	return np.mean(np.square(true_vals - pred_vals))

class Enformer:
	def __init__(self, tfhub_url):
		self._model = hub.load(tfhub_url).model

	def predict_on_batch(self, inputs):
		predictions = self._model.predict_on_batch(inputs)
		return {k: v.numpy() for k, v in predictions.items()}

	@tf.function
	def contribution_input_grad(self, input_sequence,
							  target_mask, output_head='human'):
		input_sequence = input_sequence[tf.newaxis]

		target_mask_mass = tf.reduce_sum(target_mask)
		with tf.GradientTape() as tape:
			tape.watch(input_sequence)
			prediction = tf.reduce_sum(
			  target_mask[tf.newaxis] *
			  self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

			input_grad = tape.gradient(prediction, input_sequence) * input_sequence
			input_grad = tf.squeeze(input_grad, axis=0)
		return tf.reduce_sum(input_grad, axis=-1)


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


class EnformerPeakDataset(keras.utils.Sequence):
	def __init__(self, peak_file, bigwig, genome_fa, chroms, batch_size, seq_len=393216, peak_size=2114, output_size=1024, peak_output_size=1000):
		self.removed_seqs = 0
		self.chroms = chroms 
		self.seq_offset = (seq_len - peak_size) // 2
		self.output_offset = (output_size - peak_output_size) // 2
		self.seq_len = seq_len
		self.output_size = output_size
		self.bigwig = pyBigWig.open(bigwig)
		self.genome = pyfaidx.Fasta(genome_fa, sequence_always_upper=True)
		self.batch_size = batch_size
		self.peak_table = self.load_peak_table(peak_file)
		self.seqs, self.values = self.load_seqs_and_signal()


	def load_peak_table(self, peak_file):
		peak_table = pd.read_csv(peak_file, sep="\t")
		if self.chroms is not None:
			peak_table = peak_table.loc[peak_table["chr"].isin(self.chroms)].reset_index()
		peak_table["input_start"] -= self.seq_offset
		peak_table["input_end"] += self.seq_offset
		peak_table["elem_start"] -= self.output_offset
		peak_table["elem_end"] += self.output_offset
		peak_table["elem_relative_start"] += (self.seq_offset - self.output_offset)
		peak_table["elem_relative_end"] += (self.seq_offset + self.output_offset)
		return peak_table

	def load_seqs_and_signal(self):
		dna_seqs, bigwig_data = [], []
		for peak_idx in range(len(self.peak_table)):
			chrom, start, end, out_start, out_end = self.peak_table.loc[peak_idx, "chr"], self.peak_table.loc[peak_idx, "input_start"], self.peak_table.loc[peak_idx, "input_end"], self.peak_table.loc[peak_idx, "elem_start"], self.peak_table.loc[peak_idx, "elem_end"]
			seq = self.genome[chrom][start:end].seq
			if len(seq) != self.seq_len:
				self.removed_seqs += 1
				continue
			dna_seqs.append(seq)
			bigwig_data.append(np.nan_to_num(self.bigwig.values(chrom, int(out_start), int(out_end))))

		# seqs_array, bigwig_array = dna_to_one_hot(dna_seqs), np.array(bigwig_data)
		# assert list(seqs_array.shape) == [len(self.peak_table)-self.removed_seqs, self.seq_len, 4]
		# assert list(bigwig_array.shape) == [len(self.peak_table)-self.removed_seqs, self.output_size]
		return dna_seqs, bigwig_data

		# return seqs_array, bigwig_array

	def __len__(self):
		return math.ceil((len(self.peak_table) - self.removed_seqs) / self.batch_size)

	def __getitem__(self, idx):
		idx_start = self.batch_size * idx 
		idx_end = min(len(self.peak_table), self.batch_size*(idx+1))
		one_hot_seqs = dna_to_one_hot(self.seqs[idx_start:idx_end])
		bw_vals = np.array(self.values[idx_start:idx_end])
		return one_hot_seqs, np.log(1+bw_vals.sum(-1, keepdims=True))


def predict_on_batch(model, seqs, head):
	seq_for_predict = tf.convert_to_tensor(seqs.squeeze(), tf.float32)
	# seq_for_predict = tf.convert_to_tensor(seqs, tf.float32)
	if len(seq_for_predict.shape) == 2:
		seq_for_predict = tf.expand_dims(seq_for_predict, axis=0)
	preds = model.predict_on_batch(seq_for_predict)["human"][...,head]
	central_bin = preds.shape[-1] // 2
	preds_sums = preds[...,central_bin-4:central_bin+4].sum(-1).flatten()
	return np.log(1 + preds_sums)
	# return preds_sums

def get_counts_predictions(model, peak_dataset, head):
	true_counts, pred_counts = [], []
	num_batches = len(peak_dataset)
	for idx in tqdm(range(num_batches)):
		seqs, counts = peak_dataset[idx]
		preds = predict_on_batch(model, seqs, head)
		true_counts.extend(counts)
		pred_counts.extend(preds)
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



