import pandas as pd 
import numpy as np
import argparse
np.random.seed(0)


def parse_args():
	parser = argparse.ArgumentParser(description="Given a narrowPeak file, pads them to a specified length around the summit and returns a new file")
	parser.add_argument("--input_bed", type=str, required=True, help="Bed file containing input elements")
	parser.add_argument("--output_file", type=str, help="Output file")
	parser.add_argument("--input_size", type=int, default=2114, help="Input size to the model")
	parser.add_argument("--eval_size", type=int, default=1000, help="Central portion over which embeddings will be calculated")
	args = parser.parse_args()
	return args


def make_region_df(ccre_table, args):
	region_info = {
		"chr": [],
		"input_start": [],
		"input_end": [],
		"elem_start": [],
		"elem_end": [],
		"elem_relative_start": [],
		"elem_relative_end": [],
	}

	for reg in range(len(ccre_table)):
		# if ccre_table.loc[reg, 0] == "chrM":
		# 	continue
		chrom = ccre_table.loc[reg, 0]
		summit_pos = ccre_table.loc[reg, 1] + ccre_table.loc[reg, 9]
		region_info["chr"].append(chrom)
		region_info["input_start"].append(summit_pos - args.input_size // 2)
		region_info["input_end"].append(summit_pos + args.input_size // 2)
		region_info["elem_start"].append(summit_pos - args.eval_size // 2)
		region_info["elem_end"].append(summit_pos + args.eval_size // 2)
		#Add relative positions of cCRE to table
		region_info["elem_relative_start"].append(args.input_size // 2 - args.eval_size // 2)
		region_info["elem_relative_end"].append(args.input_size // 2 + args.eval_size // 2)
	

	return pd.DataFrame(region_info)

def main():
	args = parse_args()
	regions_table = pd.read_csv(args.input_bed, sep="\t", header=None)
	model_input_df = make_region_df(regions_table, args)
	model_input_df.to_csv(args.output_file, sep="\t", header=True, index=False)

if __name__ == "__main__":
	main()
