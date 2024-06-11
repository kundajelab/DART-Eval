# DART-Eval: A Comprehensive DNA Language Model Evaluation Benchmark on Regulatory DNA

The Dart-Eval preprint is available here:
(Insert Preprint Link)

## Data

All data is available for download at Synapse project [`syn59522070`](https://www.synapse.org/Synapse:syn59522070).

## Tasks

The commands in this section reproduce the results for each task in the paper. The output files mirror the structure of the Synapse project.

### Preliminaries

Prior to running analyses, set the `$DART_WORK_DIR` environment variable. This directory will be used to store intermediate files and results.

Additionally, download the genome reference files from [`syn60581044`](https://www.synapse.org/Synapse:syn60581044) into `$DART_WORK_DIR/refs`, keeping the file names. These genome references are used across all tasks.

### Task 1: Prioritizing Known Regulatory Elements

#### Inputs

This task utilizes the set of ENCODE v3 candidate cis-regulatory elements (cCREs). A BED-format file of cCRE genomic coordinates is available at [`TODO`]. This file should be downloaded to `TODO`.

#### Dataset Generation

````bash
python -m dnalm_bench.task_1_paired_control.dataset_generators.encode_ccre --ccre_bed $DART_WORK_DIR/TODO --output_file $DART_WORK_DIR/task_1_ccre/processed_inputs/ENCFF420VPZ_processed.tsv
````

This script expands each element to 350 bp, centered on the midpoint of the element. The output file is a TSV with the following columns:

- `chrom`: chromosome
- `input_start`: start position of the length-expanded element
- `input_end`: end position of the length-expanded element
- `ccre_start`: start position of the original cCRE
- `ccre_end`: end position of the original cCRE
- `ccre_relative_start`: start position of the original cCRE relative to the length-expanded element
- `ccre_relative_end`: end position of the original cCRE relative to the length-expanded element
- `reverse_complement`: 1 if the element is reverse complemented, 0 otherwise

#### Zero-shot likelihood analyses

```bash
python -m dnalm_bench.task_1_paired_control.zero_shot.encode_ccre.$MODEL
```

#### Ab-initio models

Extract final-layer embeddings 

```bash
python -m dnalm_bench.task_1_paired_control.supervised.encode_ccre.extract_embeddings.probing_head_like
```

Train probing_head_like ab-initio model

```bash
python -m dnalm_bench.task_1_paired_control.supervised.encode_ccre.ab_initio.probing_head_like
```

Evaluate probing_head_like ab-initio model

```bash
python -m dnalm_bench.task_1_paired_control.supervised.encode_ccre.eval_ab_initio.probing_head_like
```

#### Probing models

Extract final-layer embeddings from each model

```bash
python -m dnalm_bench.task_1_paired_control.supervised.encode_ccre.extract_embeddings.$MODEL
```

Train probing models

```bash
python -m dnalm_bench.task_1_paired_control.supervised.encode_ccre.train_classifiers.$MODEL
```

Evaluate probing models

```bash
python -m dnalm_bench.task_1_paired_control.supervised.encode_ccre.eval_finetune.$MODEL $CHECKPOINT_NUM
```

where `$CHECKPOINT_NUM` is the number of the checkpoint to evaluate.

#### Finetuning models

Train finetuning models

```bash
python -m dnalm_bench.task_1_paired_control.supervised.encode_ccre.finetune.$MODEL
```

Evaluate finetuning models

```bash
python -m dnalm_bench.task_1_paired_control.supervised.encode_ccre.eval_finetune.$MODEL $CHECKPOINT_NUM
```

where `$CHECKPOINT_NUM` is the number of the checkpoint to evaluate


### Task 2: Transcription Factor Motif Footprinting

#### Dataset Generation:

```bash
python -m dnalm_bench.task_2_5_single.dataset_generators.transcription_factor_binding.h5_to_seqs.py [INPUT EMBEDDING FILE]
python -m dnalm_bench.task_2_5_single.dataset_generators.motif_footprinting_dataset --input_seqs [INPUT_SEQS] --output_file [OUTPUT_FILE] --meme_file [MEME_MOTIF_FILE]
```

#### Extracting Embeddings: 

```bash
python -m dnalm_bench.task_2_5_single.experiments.task_2_transcription_factor_binding.embeddings.$MODEL
```
#### Likelihoods:
```bash
python -m dnalm_bench.task_2_5_single.experiments.task_2_transcription_factor_binding.likelihoods.$MODEL
```
#### Eval Metrics Calculation
```bash
python -m dnalm_bench.task_2_5_single.experiments.task_2_transcription_factor_binding.footprint_eval_likelihoods.py --input_seqs [INPUT_DATASET] --likelihoods [LIKELIHOODS] --ouput_file [OUTPUT_FILE]
python -m dnalm_bench.task_2_5_single.experiments.task_2_transcription_factor_binding.footprint_eval_embeddings.py --input_seqs [INPUT_DATASET] --embeddings [EMBEDDINGS] --ouput_file [OUTPUT_FILE]
```
#### Further Evaluation Notebooks

```dnalm_bench/task_2_5_single/experiments/eval_footprinting_likelihood.ipynb``` - figure production for likelihood-based evaluation
```dnalm_bench/task_2_5_single/experiments/eval_footprinting_embedding.ipynb``` - figure production for embedding-based evaluation
```dnalm_bench/task_2_5_single/experiments/footprinting_pairwise.ipynb``` - cross-model pairwise production plots
```dnalm_bench/task_2_5_single/experiments/footprinting_conf_intervals.ipynb``` - confidence interval calculation

### Task 3: Discriminating Cell-Type-Specific Elements

#### Dataset Generation:

Using the input peaks from ENCODE, generate a consensus peakset:\
python dnalm_bench.task_2_5_single.dataset_generators.peak_classification.make_consensus_peakset.py

Then, generate individual counts matrices for each sample, using the bam files downloaded from ENCODE and the consensus peakset:
python dnalm_bench.task_2_5_single.dataset_generators.peak_classification.generate_indl_counts_matrix.py

Concatenate the counts matrices and generate DESeq inputs:\
python dnalm_bench.task_2_5_single.dataset_generators.peak_classification.generate_merged_counts_matrix.py

Finally, run DESeq for each cell type to obtain differentially accessible peaks for each cell type:\
dnalm_bench.task_2_5_single.dataset_generators.peak_classification.DESeqAtac.R

You will end up with the file: (INSERT Synapse link)

#### Extracting Embeddings:

python dnalm_bench.task_2_5_single.experiments.task_3_peak_classification.extract_embeddings.$MODEL $CELL_LINE $CATEGORY

````
CELL_LINE:
CATEGORY:
````

#### Clustering:

python dnalm_bench.task_2_5_single.experiments.task_3_peak_classification.cluster [EMBEDDING FILE] [LABEL FILE] [INDEX FILE] [OUT_DIR]

#### Training:

_Probed_: python -m dnalm_bench.task_2_5_single.experiments.task_3_peak_classification.train.$MODEL\
_Finetuned_: python -m dnalm_bench.task_2_5_single.experiments.task_3_peak_classification.finetune.$MODEL

#### Evals:

_Probed_: python -m dnalm_bench.task_2_5_single.experiments.task_3_peak_classification.eval_probing.$MODEL\
_Finetuned_: python -m dnalm_bench.task_2_5_single.experiments.task_3_peak_classification.eval_finetune.$MODEL

### Task 4: Predicting Chromatin Activity from Sequence


### Task 5: Chromatin Activity Variant Effect Prediction

#### Zero-Shot:
##### Embeddings: 
```bash
python -m dnalm_bench.task_2_5_single.experiments.task_5_variant_effect_prediction.zero_shot_embeddings.$MODEL $VARIANTS_BED $OUTPUT_PREFIX $GENOME_FA
```

##### Likelihoods:
```bash
python -m dnalm_bench.task_2_5_single.experiments.task_5_variant_effect_prediction.zero_shot_likelihoods.$MODEL $VARIANTS_BED $OUTPUT_PREFIX $GENOME_FA
```

#### Probed:
```bash
python -m dnalm_bench.task_2_5_single.experiments.task_5_variant_effect_prediction.probed_log_counts.$MODEL $VARIANTS_BED $OUTPUT_PREFIX $GENOME_FA
```

#### Finetuned:
```bash
python -m dnalm_bench.task_2_5_single.experiments.task_5_variant_effect_prediction.finetuned_log_counts.$MODEL $VARIANTS_BED $OUTPUT_PREFIX $GENOME_FA
```

#### Evaluation Notebooks

Helper functions called in the evaluation notebooks: ```dnalm_bench.task_2_5_single.experiments.task_5_variant_effect_prediction.variant_tasks.py```

Zero Shot Evaluation Notebook: ```dnalm_bench.task_2_5_single.experiments.task_5_variant_effect_prediction.Zero_Shot_Final.ipynb```

Probed Evaluation Notebook: ```dnalm_bench.task_2_5_single.experiments.task_5_variant_effect_prediction.Probed_Final_Counts.ipynb```

Finetuned Evaluation Notebook: ```dnalm_bench.task_2_5_single.experiments.task_5_variant_effect_prediction.Finetuned_Final_Counts.ipynb```
