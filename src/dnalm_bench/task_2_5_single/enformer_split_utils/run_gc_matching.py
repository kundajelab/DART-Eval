import argparse
import os
import pandas as pd


def parse_args():
	parser = argparse.ArgumentParser(description="Collects arguments to run GC matching using the ChromBPNet Multitask repo")
	parser.add_argument("--peak_file", type=str)
	parser.add_argument("--chrom_size_file", type=str)
	parser.add_argument("--reference_genome", type=str)
	parser.add_argument("--blacklist", type=str)
	parser.add_argument("--output_dir", type=str)
	parser.add_argument("--input_len", type=int, default=2114)
	parser.add_argument("--stride", type=int, default=1000)
	args = parser.parse_args()
	return args

#This needs to be run from the directory storing the chrompbnet repo
def run_matching(peak_file, chrom_size_file, genome_file, output_dir, blacklist, input_len, stride):
	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)
	bins_output_prefix = output_dir + "genomewide_gc_bins"
	print("Getting Genome-Wide GC Bins")
	get_bins_command = f"python src/helpers/make_gc_matched_negatives/get_genomewide_gc_buckets/get_genomewide_gc_bins.py \
	-g {genome_file} -o {bins_output_prefix} -f {input_len} -s {stride}"
	os.system(get_bins_command)
	bins_file = output_dir + "genomewide_gc_bins.bed"
	
	print("Generating GC-Matched Negatives")
	gc_match_command = f"bash step3_get_background_regions.sh {genome_file} {chrom_size_file} {blacklist} {peak_file} \
	{input_len} {bins_file} {output_dir}"
		

	os.system(gc_match_command)


def main():
	args = parse_args()
	peak_file = args.peak_file
	chrom_size_file = args.chrom_size_file
	ref_genome = args.reference_genome
	blacklist = args.blacklist
	output_dir = args.output_dir
	run_matching(peak_file, chrom_size_file, ref_genome, output_dir, blacklist, args.input_len, args.stride)

if __name__ == "__main__":
	main()