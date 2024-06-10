import pandas as pd
import glob
from collections import Counter
import os

dart_work_dir = os.environ.get("DART_WORK_DIR", "")

counts_matrices = os.path.join(dart_work_dir, "task_3_cell-type-specific/input_data/*/ENCF*.csv")

merged_df = pd.read_csv(glob.glob(counts_matrices)[0], header=0)
for f in glob.glob(counts_matrices)[1:]:
    counts_df = pd.read_csv(f, header=0)
    print(counts_df.shape)
    merged_df = pd.merge(merged_df, counts_df, on="peak")
    print(merged_df.shape)

merged_df_output_path = os.path.join(dart_work_dir, "task_3_cell-type-specific/input_data/merged_counts_matrix.csv")
merged_df.to_csv(merged_df_output_path, index=False)

sample_names = list(merged_df.columns[1:])
print(sample_names)

def create_dataframe(samples, index):
    # Extract the main sample and its cell type
    print(samples)
    print(index)
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

for i in range(0, len(sample_names), 3):
    df, cell_type = create_dataframe(sample_names, i)
    output_path = os.path.join(dart_work_dir, "task_3_cell-type-specific/input_data/", f"{cell_type}/{cell_type}_deseq_input_coldata.csv")
    df.to_csv(output_path, index=False)
