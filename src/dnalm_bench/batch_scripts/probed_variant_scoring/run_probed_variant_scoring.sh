#!/bin/sh

model=$1
variantsbed=$2
countstsv=$3
genome=$4
celltype=$5


ml python/3.9.0
source /oak/stanford/groups/akundaje/patelas/sherlock_venv/chrombench-new/bin/activate


python -m dnalm_bench.single.experiments.variant_effect_prediction.probed_log_counts.$model $variantsbed $countstsv $genome $celltype