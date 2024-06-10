import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description="Given a list of motif-inserted sequences and their likelihoods, determine model performance on each motif")
	parser.add_argument("--input_seqs", type=str, required=True, help="Text file containing input negative elements which were footprinted")
	parser.add_argument("--likelihoods", type=str, required=True, help="tsv file containing model likelihoods")
	parser.add_argument("--output_file", type=str, required=True, help="Output file path containing metrics")
	args = parser.parse_args()
	return args


def get_likelihoods(combined_df):
	likelihood_dict = {}
	for i, key in enumerate(combined_df.index):
		if i % 2 == 1:
			continue
		key_split = key.split("_")
		if len(key_split) == 3:
			continue
		motif = key_split[1]
		if motif not in likelihood_dict:
			likelihood_dict[motif] = {"true": [], "shuffled": []}
		next_key = combined_df.index[i+1]
		likelihood_dict[motif]["true"].append(combined_df.loc[key, 2])
		likelihood_dict[motif]["shuffled"].append(combined_df.loc[next_key, 2])
	return likelihood_dict


def get_accuracies(likelihood_dict):
	accuracy_dict = {}
	for motif in likelihood_dict:
		curr_dict = likelihood_dict[motif]
		num_seqs = len(curr_dict["true"])
		diffs = [curr_dict["true"][x] - curr_dict["shuffled"][x] for x in range(num_seqs)]
		accuracy_dict[motif] = np.mean([x >= 0 for x in diffs])
	return accuracy_dict


def get_pvals(likelihood_dict):
	pval_dict = {}
	for motif in likelihood_dict:
		curr_dict = likelihood_dict[motif]
		pval = wilcoxon(curr_dict["true"], curr_dict["shuffled"], alternative="greater")[1]
		pval_dict[motif] = pval
	return pval_dict


def get_background_accs(combined_df, likelihood_dict):
	background_acc_dict = {}
	true_likelihoods = np.array((combined_df.filter(regex="raw", axis=0)[2]))
	for motif in likelihood_dict:
		motiflik = likelihood_dict[motif]
		true_diffs = np.abs(true_likelihoods - np.array(motiflik["true"]))
		shuf_diffs = np.abs(true_likelihoods - np.array(motiflik["shuffled"]))
		background_acc_dict[motif] = np.mean(true_diffs > shuf_diffs)
	return background_acc_dict


def main():
	args = parse_args()
	seq_data = pd.read_csv(args.input_seqs, sep="\t", header=None)
	likelihood_data = pd.read_csv(args.likelihoods, sep="\t", header=None)
	combined_df = pd.concat([seq_data, likelihood_data], axis=1, ignore_index=True)
	combined_df = combined_df.set_index(0, drop=True)
	likelihood_dict = get_likelihoods(combined_df)
	accuracies = get_accuracies(likelihood_dict)
	pvals = get_pvals(likelihood_dict)
	background_accs = get_background_accs(combined_df, likelihood_dict)
	final_df = pd.DataFrame([accuracies, pvals, background_accs], index=["Accuracy", "P-Value", "Accuracy vs Background"]).T
	final_df.to_csv(args.output_file, sep="\t", index=True, header=True)


if __name__ == "__main__":
	main()
