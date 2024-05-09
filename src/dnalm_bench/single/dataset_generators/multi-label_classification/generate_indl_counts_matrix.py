import pysam
import pandas as pd
import sys

# Load the peaks from a BED file
# peaks_file = '/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/accepted_peaks.tsv'
peaks_file = '/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/old_da_peaks.tsv'

data_types = {
    'chrom': 'str',
    'start': 'int64',
    'end': 'int64',
}

peaks_df = pd.read_csv(peaks_file, sep='\t', header=0, names=['chrom', 'start', 'end'], dtype=data_types)

# Open the BAM file

cell_type = sys.argv[1]
file_name = sys.argv[2]
bam_path = f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/{cell_type}/{file_name}.bam"
bam_file = pysam.AlignmentFile(bam_path, "rb")

# Function to count reads in a given peak
def count_reads_in_peak(bam, chrom, start, end):
    count = 0
    for read in bam.fetch(chrom, start, end):
        if not read.is_unmapped:  # Skip unmapped reads
            count += 1
    return count

# Apply the counting function to each peak
peaks_df['read_count'] = peaks_df.apply(lambda row: count_reads_in_peak(bam_file, row['chrom'], row['start'], row['end']), axis=1)

peaks_df.rename(columns={'read_count': f'{cell_type}_{file_name}'}, inplace=True)
peaks_df['peak'] = peaks_df.apply(lambda row: f"{row['chrom']}:{row['start']}-{row['end']}", axis=1)

# Optionally, save the results to a new CSV file
peaks_df[["peak", f'{cell_type}_{file_name}']].to_csv(f'/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/{cell_type}/{file_name}.csv', index=False)

# Close the BAM file
bam_file.close()
