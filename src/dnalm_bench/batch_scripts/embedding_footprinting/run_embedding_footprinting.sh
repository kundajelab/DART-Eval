#!/bin/sh

model=$1

ml python/3.9.0
source /oak/stanford/groups/akundaje/patelas/sherlock_venv/chrombench-new/bin/activate


python -m dnalm_bench.single.experiments.footprinting.embeddings.$model