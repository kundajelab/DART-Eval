import pandas as pd
import glob
from collections import Counter

counts_matrices = "/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/*/*.csv"

merged_df = pd.read_csv(glob.glob(counts_matrices)[0], header=0)
for f in glob.glob(counts_matrices)[1:]:
    counts_df = pd.read_csv(f, header=0)
    print(counts_df.shape)
    merged_df = pd.merge(merged_df, counts_df, on="peak")
    print(merged_df.shape)

merged_df.to_csv("/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/merged_counts_matrix.csv", index=False)

sample_names = list(merged_df.columns[1:])

def create_dataframe(samples, index):
    # Extract the main sample and its cell type
    main_sample = samples[index]
    main_cell_type = main_sample.split('_')[0]
    print(main_cell_type)

    # Create the DataFrame
    df = pd.DataFrame({
        'names': samples,
        'condition': ['other' if sample.split('_')[0] != main_cell_type else main_cell_type for sample in samples]
    })
    df["type"]="paired-end"

    return df, main_cell_type

for i in range(0, 14, 3):
    df, cell_type = create_dataframe(sample_names, i)
    df.to_csv(f"/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/{cell_type}_deseq_input_coldata.csv", index=False)
