import pandas as pd 
import numpy as np
import argparse
import pyfaidx
import random
np.random.seed(0)


def parse_args():
	parser = argparse.ArgumentParser(description="Given a set of negative regions, ")
	parser.add_argument("--input_seqs", type=str, required=True, help="Text file containing input negative elements which will be footprinted")
	parser.add_argument("--output_file", type=str, required=True, help="Output file")
	parser.add_argument("--meme_file", type=str, required=True, help="Meme file containing motif PWMs")
	args = parser.parse_args()
	return args

def shuffle(seq):
	seq_list = list(seq)
	orig_seq_list = seq_list.copy()
	while seq_list == orig_seq_list:
		random.shuffle(seq_list)
	assert seq_list != orig_seq_list
	return ''.join(seq_list)

def pwm_to_consensus(pwm):
	seq_to_base = {0:"A", 1:"C", 2:"G", 3:"T"}
	return "".join([seq_to_base[x] for x in pwm.argmax(1)])

def reverse_complement(seq):
	seq = seq.replace("A", "t").replace("C", "g").replace("T", "a").replace("G", "c").upper()
	return seq[::-1]

def read_meme(filename):
	motifs = {}

	with open(filename, "r") as infile:
		motif, width, i = None, None, 0

		for line in infile:
			if motif is None:
				if line[:5] == 'MOTIF':
					motif = line.split()[1]
				else:
					continue

			elif width is None:
				if line[:6] == 'letter':
					width = int(line.split()[5])
					pwm = np.zeros((width, 4))

			elif i < width:
				pwm[i] = list(map(float, line.split()))
				i += 1

			else:
				motifs[motif] = pwm_to_consensus(pwm)
				motif, width, i = None, None, 0

	return motifs

def insert_motif(seq, motif):
	'''
	Inserts a motif into the middle of a given sequence (replaces the original middle nucleotides)
	'''
	motif_len = len(motif)
	insert_loc = len(seq) // 2 - len(motif) // 2
	seq_w_motif = seq[:insert_loc] + motif + seq[insert_loc + len(motif):]
	assert len(seq_w_motif) == len(seq)
	return seq_w_motif

def get_all_inserts_for_seq(seq, motif_dict):
	'''
	Given a particular sequence, inserts all motifs into the sequence separately and returns the compiled dict
	In addition to the true motif, also inserts a shuffled version of each motif
	'''
	seq_dict = {}
	for motif in motif_dict:
		seq_dict[motif] = [insert_motif(seq, motif_dict[motif]), insert_motif(seq, shuffle(motif_dict[motif]))]
	return seq_dict

def compile_seqs(seq_file, motif_dict):
	'''
	Obtains all insertions for all sequences
	In this step, we also take the reverse complement of all sequences
	'''
	overall_seq_dict = {}
	seq_file_obj = open(seq_file, "r")
	for row, line in enumerate(seq_file_obj):
		seq = line.strip()
		overall_seq_dict[str(row) + "_" + "raw" + "_forward"] = seq
		overall_seq_dict[str(row) + "_" + "raw" + "_reverse"] = reverse_complement(seq)
		curr_seq_dict = get_all_inserts_for_seq(seq, motif_dict)
		for motif in curr_seq_dict:
			overall_seq_dict[str(row) + "_" + motif + "_forward_true"] = curr_seq_dict[motif][0]
			overall_seq_dict[str(row) + "_" + motif + "_forward_shuffled"] = curr_seq_dict[motif][1]
			overall_seq_dict[str(row) + "_" + motif + "_reverse_true"] = reverse_complement(curr_seq_dict[motif][0])
			overall_seq_dict[str(row) + "_" + motif + "_reverse_shuffled"] = reverse_complement(curr_seq_dict[motif][1])
	return overall_seq_dict


def write_to_file(overall_seq_dict, out_file):
	file_out = open(out_file, "w")
	for seq in overall_seq_dict:
		file_out.write(seq + "\t" + overall_seq_dict[seq] + "\n")
	file_out.close()


def main():
	args = parse_args()
	motif_dict = read_meme(args.meme_file)
	inserted_seqs_dict = compile_seqs(args.input_seqs, motif_dict)
	write_to_file(inserted_seqs_dict, args.output_file)


if __name__ == "__main__":
	main()