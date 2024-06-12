import numpy as np
import pandas as pd
import os
import subprocess

root_output_dir = os.environ.get("DART_WORK_DIR", "")

def initialize_motif_table(peak_data, motif_data):
	motif_array = np.zeros([len(peak_data), len(motif_data)])
	table_index = [str((peak_data.loc[x, "chr"], peak_data.loc[x, "input_start"], peak_data.loc[x, "input_end"])) for x in range(len(peak_data))]
	motif_table = pd.DataFrame(motif_array, index=table_index, columns=motif_data.index)
	return motif_table 


def load_fimo_results(fimo_file_list):
	fimo_tables = [pd.read_csv(x, sep="\t", header=None) for x in fimo_file_list]
	fimo_table_overall = pd.concat(fimo_tables, ignore_index=True)
	return fimo_table_overall


def populate_hits(hit_table, fimo_results):
	for hit in range(len(fimo_results)):
	    peak_loc = (fimo_results.loc[hit, 0], fimo_results.loc[hit, 1], fimo_results.loc[hit, 2])
	    motif_id = fimo_results.loc[hit, 6]
	    hit_table.loc[str(peak_loc), motif_id] += 1

	return hit_table


def main():
	print("Loading peak data")
	peak_file = os.path.join(root_output_dir, f"task_3_cell-type-specific/processed_inputs/peaks_by_cell_label_unique_dataloader_format.tsv")
	peak_data = pd.read_csv(peak_file, sep="\t")

	print("Loading motif data")
	motif_family_file = os.path.join(root_output_dir, f"task_2_footprinting/input_data/H12CORE_motifs.tsv")
	motif_family_data = pd.read_csv(motif_family_file, sep="\t", index_col=0)

	print("Loading FIMO results")
	fimo_base_dir = os.path.join(root_output_dir, f"task_3_cell-type-specific/processed_inputs/fimo/")
	cell_line_list = ["K562", "GM12878", "HEPG2", "IMR90", "H1ESC"]
	fimo_file_list = []
	for cell_line in cell_line_list:
		fimo_file_list.append(fimo_base_dir + cell_line + "/fimo_out/fimo_all_hits_intersect.tsv")

	fimo_results = load_fimo_results(fimo_file_list)
	print(fimo_results.head())

	motif_table_empty = initialize_motif_table(peak_data, motif_family_data)
	print(motif_table_empty.head())

	print("Populating output data")
	motif_table_filled = populate_hits(motif_table_empty, fimo_results)

	print("Saving output file")
	motif_table_filled.to_csv(os.path.join(root_output_dir, f"task_3_cell-type-specific/processed_inputs/fimo/motif_count_matrix_total_hits.tsv"), sep="\t", header=True, index=True)


if __name__ == "__main__":
	main()
