import h5py
import numpy as np
import sys


#This script is used with the footprinting task - it takes an embedding file with positives and controls, extracts 100 controls, converts them to DNA sequences, and saves that into a text file

onehot_h5 = sys.argv[1]
onehot_data = h5py.File(onehot_h5, "r")

def pwm_to_consensus(pwm):
	seq_to_base = {0:"A", 1:"C", 2:"G", 3:"T"}
	return "".join([seq_to_base[x] for x in pwm.argmax(1)])


seqs = []
key_list = list(onehot_data['ctrl'].keys())
for key_ind in list(range(0, len(key_list), 20))[:100]:
    seqs.append(onehot_data['ctrl'][key_list[key_ind]][567])


out_file = sys.argv[2]
file_obj = open(out_file, "w")
for seq in seqs:
    file_obj.write(pwm_to_consensus(seq) + "\n")

