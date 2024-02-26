import pandas as pd 
import numpy as np
import argparse
np.random.seed(0)


def parse_args():
	parser = argparse.ArgumentParser(description="Given a bed file of ccres, jitters them to produce input regions")
	parser.add_argument("--ccre_bed", type=str, required=True, help="Bed file containing ccre regions")
	parser.add_argument("--output_file", type=str, help="Output file")
	parser.add_argument("--input_size", type=int, default=2114, help="Input size to the model")
	parser.add_argument("--max_jitter", type=int, default=500, help="Max jitter on either side")
	args = parser.parse_args()
	return args


def make_region_df(ccre_table, args):
	region_info = {
		"chr": [],
		"input_start": [],
		"input_end": [],
		"ccre_start": [],
		"ccre_end": [],
		"ccre_relative_start": [],
		"ccre_relative_end": [],
		"reverse_complement": []
	}

	for reg in range(len(ccre_table)):
		region_info["chr"].append(ccre_table.loc[reg, 0])
		start, end = ccre_table.loc[reg, 1], ccre_table.loc[reg, 2]
		region_info["ccre_start"].append(start)
		region_info["ccre_end"].append(end)
		#Determine how much to expand the region by
		length = end - start
		to_expand = args.input_size + 2 * args.max_jitter - length
		expand_start = start - to_expand // 2
		#Calculate final start and end positions of input region to model
		jitter_start_pos = np.random.randint(expand_start, expand_start + 2 * args.max_jitter)
		jitter_end_pos = jitter_start_pos + args.input_size
		#Add model start and end positions to table
		region_info["input_start"].append(jitter_start_pos)
		region_info["input_end"].append(jitter_end_pos)
		#Add relative positions of cCRE to table
		region_info["ccre_relative_start"].append(start - jitter_start_pos)
		region_info["ccre_relative_end"].append(end - jitter_start_pos)
		#Add whether to reverse-complement the sequence
		region_info["reverse_complement"].append(np.random.choice([True, False]))
	

	return pd.DataFrame(region_info)

def main():
	args = parse_args()
	ccre_table = pd.read_csv(args.ccre_bed, sep="\t", header=None)
	model_input_df = make_region_df(ccre_table, args)
	model_input_df.to_csv(args.output_file, sep="\t", header=True, index=False)

if __name__ == "__main__":
	main()
