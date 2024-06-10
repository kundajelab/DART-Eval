import numpy as np
import argparse
import pandas as pd
np.random.seed(0)

def parse_args():
	parser = argparse.ArgumentParser(description="Given a file of differentially accessible peaks, changes the format to the desired one")
	parser.add_argument("--input_file", type=str, required=True, help="File with differentially accessible peaks for each dataset")
	parser.add_argument("--output_file", type=str, help="Output file")
	args = parser.parse_args()
	return args


def reformat(region_table):
	table_final = region_table.copy()
	table_final["input_start"] = table_final["start"]
	table_final["input_end"] = table_final["end"]
	table_final["elem_start"] = table_final["start"]
	table_final["elem_end"] = table_final["end"]
	table_final["chr"] = table_final["chrom"]
	table_final["is_peak"] = True
	table_final = table_final[["chr", "input_start", "input_end", "elem_start", "elem_end", "is_peak", "label"]]
	return table_final


def main():
	args = parse_args()
	region_table = pd.read_csv(args.input_file, sep="\t")
	final_table = reformat(region_table)
	final_table.to_csv(args.output_file, sep="\t", header=True, index=False)

if __name__ == "__main__":
	main()
