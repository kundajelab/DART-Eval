# DART-Eval: A Comprehensive DNA Language Model Evaluation Benchmark on Regulatory DNA

The Dart-Eval preprint is available here:
(Insert Preprint Link)

## Data
All data is available for download [here](INSERT Synapse link)

## Tasks

### Task 1: Prioritizing Known Regulatory Elements

#### Dataset Generation:
`python -m dnalm_bench.paired_control.dataset_generators.encode_ccre`

#### Extracting Embeddings: 
`python -m dnalm_bench.paired_control.supervised.encode_ccre.extract_embeddings.$MODEL`

#### Training:
_Probed_: `python -m dnalm_bench.paired_control.supervised.encode_ccre.train_classifiers.$MODEL` \
_Finetuned_: `python -m dnalm_bench.paired_control.supervised.encode_ccre.finetune.$MODEL`

#### Evals:
_Probed_: `python -m dnalm_bench.paired_control.supervised.encode_ccre.eval_probing.$MODEL` \
_Finetuned_: `python -m dnalm_bench.paired_control.supervised.encode_ccre.eval_finetune.$MODEL`

### Task 2: Transcription Factor Motif Footprinting

### Task 3: Discriminating Cell-Type-Specific Elements

### Task 4: Predicting Chromatin Activity from Sequence

### Task 5: Chromatin Activity Variant Effect Prediction
