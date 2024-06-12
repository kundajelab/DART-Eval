import pysam
import pandas as pd
import sys
import os

dart_work_dir = os.environ.get("DART_WORK_DIR", "")

# Load the peaks from a BED file
peaks_file = os.path.join(dart_work_dir,"task_3_peak_classification/input_data/accepted_peaks_all.tsv")

data_types = {
    'chrom': 'str',
    'start': 'int64',
    'end': 'int64',
}

peaks_df = pd.read_csv(peaks_file, sep='\t', header=0, names=['chrom', 'start', 'end'], dtype=data_types)

cell_type = sys.argv[1]
file_name = sys.argv[2]
bam_path = os.path.join(dart_work_dir,f"task_3_peak_classification/input_data/{cell_type}/{file_name}.bam")
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
output_path = os.path.join(dart_work_dir, "task_3_peak_classification/input_data", f"{cell_type}/{file_name}.csv")
peaks_df[["peak", f'{cell_type}_{file_name}']].to_csv(output_path, index=False)

# Close the BAM file
bam_file.close()
