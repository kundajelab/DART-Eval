import pandas as pd

import os
import glob

###
# HELPER FUNCTIONS
###
def load_peaks(peak_file):
	peak_df = pd.read_csv(peak_file, sep="\t", names=["chrom", "start", "end", "name", "score", "strand", "signal", "p", "q", "summit"], compression='gzip')
	return peak_df

def add_peak(accepted_peaks_dict, accepted_peaks_df, new_peak):
	np_chro, np_start, np_end = new_peak
	if np_chro in accepted_peaks_dict:
		accepted_peaks_dict_chro = accepted_peaks_dict[np_chro]
		peak_unique = True
		for p in accepted_peaks_dict_chro:
			if peak_overlap(p, (np_start, np_end)):
				peak_unique = False
				break
		if peak_unique:
			accepted_peaks_dict[np_chro] += [(np_start, np_end)]
			accepted_peaks_df = pd.concat([accepted_peaks_df, pd.DataFrame({"chrom": np_chro, "start": np_start, "end": np_end}, index=[0])], ignore_index=True)
	else:
		accepted_peaks_dict[np_chro] = [(np_start, np_end)]
		accepted_peaks_df = pd.concat([accepted_peaks_df, pd.DataFrame({"chrom": np_chro, "start": np_start, "end": np_end}, index=[0])], ignore_index=True)
	return accepted_peaks_dict, accepted_peaks_df

def peak_overlap(peaka, peakb):
	return ((peakb[0] < peaka[0]) and (peakb[1] >= peaka[0])) or ((peakb[0] >= peaka[0]) and (peakb[0] < peaka[1]))


###
# CODE START
###

peaks_dir = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/*/*narrowPeak.gz"
peaks = []

for f in glob.glob(peaks_dir):
	print(f)
	peaks.append(load_peaks(f))

peaks_combined = pd.concat(peaks, ignore_index=True)
peaks_combined = peaks_combined.sort_values("q", ascending=False)
peaks_combined = peaks_combined.reset_index(drop=True)

accepted_peaks_dict = {}
accepted_peaks_df = pd.DataFrame()
for index, row in peaks_combined.iterrows():
	if index%10000 == 0:
		print(index)
	p_summit = row["start"] + row["summit"]
	p = (row["chrom"], p_summit - 250, p_summit + 250)
	accepted_peaks_dict, accepted_peaks_df = add_peak(accepted_peaks_dict, accepted_peaks_df, p)
accepted_peaks_df = accepted_peaks_df.sort_values(by=["chrom", "start"], ascending=[True, True])
accepted_peaks_df = accepted_peaks_df.reset_index(drop=True)

out_loc = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/accepted_peaks.tsv"

accepted_peaks_df.to_csv(out_loc, sep="\t", index=False)

