# DART-Eval: A Comprehensive DNA Language Model Evaluation Benchmark on Regulatory DNA

The Dart-Eval preprint is available here:
(Insert Preprint Link)

## Data
All data is available for download [here](INSERT Synapse link)

## Tasks

### Task 1: Prioritizing Known Regulatory Elements

#### Dataset Generation:
python -m dnalm_bench.paired_control.dataset_generators.encode_ccre --ccre_bed [CCRE_BED] --output_file [OUTPUT_FILE]
````
--ccre_bed: $CCRE_BED
--output_file: $OUTPUT_FILE
````

To generate the exact dataset we used, set `max_jitter` to 0 and `input_size` to 350. 

#### Extracting Embeddings: 
python -m dnalm_bench.paired_control.supervised.encode_ccre.extract_embeddings.$MODEL

#### Training:
_Probed_: python -m dnalm_bench.paired_control.supervised.encode_ccre.train_classifiers.$MODEL \
_Finetuned_: python -m dnalm_bench.paired_control.supervised.encode_ccre.finetune.$MODEL

#### Evals:
_Probed_: python -m dnalm_bench.paired_control.supervised.encode_ccre.eval_probing.$MODEL \
_Finetuned_: python -m dnalm_bench.paired_control.supervised.encode_ccre.eval_finetune.$MODEL

### Task 2: Transcription Factor Motif Footprinting

#### Dataset Generation:
python -m dnalm_bench.task_2_5_single.dataset_generators.transcription_factor_binding.h5_to_seqs.py [INPUT EMBEDDING FILE]
python -m dnalm_bench.task_2_5_single.dataset_generators.motif_footprinting_dataset --input_seqs [INPUT_SEQS] --output_file [OUTPUT_FILE] --meme_file [MEME_MOTIF_FILE]

#### Extracting Embeddings: 
python -m dnalm_bench.task_2_5_single.experiments.task_2_transcription_factor_binding.embeddings.$MODEL

#### Likelihoods:
python -m dnalm_bench.task_2_5_single.experiments.task_2_transcription_factor_binding.likelihoods.$MODEL

#### Eval Metrics Calculation
python -m dnalm_bench.task_2_5_single.experiments.task_2_transcription_factor_binding.footprint_eval_likelihoods.py --input_seqs [INPUT_DATASET] --likelihoods [LIKELIHOODS] --ouput_file [OUTPUT_FILE]
python -m dnalm_bench.task_2_5_single.experiments.task_2_transcription_factor_binding.footprint_eval_embeddings.py --input_seqs [INPUT_DATASET] --embeddings [EMBEDDINGS] --ouput_file [OUTPUT_FILE]

#### Further Evaluation Notebooks
src/dnalm_bench/task_2_5_single/experiments/eval_footprinting_likelihood.ipynb - figure production for likelihood-based evaluation
src/dnalm_bench/task_2_5_single/experiments/eval_footprinting_embedding.ipynb - figure production for embedding-based evaluation
src/dnalm_bench/task_2_5_single/experiments/footprinting_pairwise.ipynb - cross-model pairwise production plots
src/dnalm_bench/task_2_5_single/experiments/footprinting_conf_intervals.ipynb - confidence interval calculation

### Task 3: Discriminating Cell-Type-Specific Elements

#### Dataset Generation:
Using the input peaks from ENCODE, generate a consensus peakset:\
python dnalm_bench.single.dataset_generators.multi-label_classification.make_consensus_peakset.py

Then, generate individual counts matrices for each sample, using the bam files downloaded from ENCODE and the consensus peakset:
python dnalm_bench.single.dataset_generators.multi-label_classification.generate_indl_counts_matrix.py

Concatenate the counts matrices and generate DESeq inputs:\
python dnalm_bench.single.dataset_generators.multi-label_classification.generate_merged_counts_matrix.py

Finally, run DESeq for each cell type to obtain differentially accessible peaks for each cell type:\
dnalm_bench.single.dataset_generators.multi-label_classification.DESeqAtac.R

You will end up with the file: (INSERT Synapse link)

#### Extracting Embeddings:
python dnalm_bench.single.experiments.cell_lines.extract_embeddings.$MODEL $CELL_LINE $CATEGORY
````
CELL_LINE:
CATEGORY:
````

#### Training:
_Probed_: python -m dnalm_bench.single.experiments.cell_lines.train.$MODEL\
_Finetuned_: python -m dnalm_bench.single.experiments.cell_lines.finetune.$MODEL

#### Evals:
_Probed_: python -m dnalm_bench.single.experiments.cell_lines.eval_probing.$MODEL\
_Finetuned_: python -m dnalm_bench.single.experiments.cell_lines.eval_finetune.$MODEL

### Task 4: Predicting Chromatin Activity from Sequence

### Task 5: Chromatin Activity Variant Effect Prediction
