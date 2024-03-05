import pandas as pd 
import numpy as np
import argparse
np.random.seed(0)


def parse_args():
	parser = argparse.ArgumentParser(description="Given a bed file DNA sequences, pads them to uniform length and returns a new file")
	parser.add_argument("--input_bed", type=str, required=True, help="Bed file containing input elements")
	parser.add_argument("--output_file", type=str, help="Output file")
	parser.add_argument("--input_size", type=int, default=2114, help="Input size to the model")
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

	unique_start_ends = set()
	for reg in range(len(ccre_table)):
		if ccre_table.loc[reg, 0] == "chrM":
			continue
		chrom = ccre_table.loc[reg, 0]
		start, end = ccre_table.loc[reg, 1], ccre_table.loc[reg, 2]
		if (chrom, start, end) in unique_start_ends:
			continue
		region_info["chr"].append(chrom)
		region_info["elem_start"].append(start)
		region_info["elem_end"].append(end)
		#Determine how much to expand the region by
		length = end - start
		# if length > args.input_size:
		# 	print(length)
		# 	assert length < args.input_size
		to_expand = args.input_size - length
		expand_start = max(0, start - to_expand // 2)
		expand_end = expand_start + args.input_size
		region_info["input_start"].append(expand_start)
		region_info["input_end"].append(expand_end)
		#Add relative positions of cCRE to table
		region_info["elem_relative_start"].append(start - expand_start)
		region_info["elem_relative_end"].append(end - expand_start)
		unique_start_ends.add((chrom, start, end))
	

	return pd.DataFrame(region_info)

def main():
	args = parse_args()
	regions_table = pd.read_csv(args.input_bed, sep="\t", header=None)
	model_input_df = make_region_df(regions_table, args)
	model_input_df.to_csv(args.output_file, sep="\t", header=True, index=False)

if __name__ == "__main__":
	main()
