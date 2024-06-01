#!/bin/sh

model=$1

export HF_HOME=/scratch/groups/akundaje/dnalm_benchmark/.cache

sbatch --export=ALL --requeue \
    -J embfootprinting.$model \
    -p akundaje,gpu,owners -t 10:00:00 \
    -G 1 -C "GPU_MEM:80GB|GPU_SKU:A100_PCIE|GPU_SKU:A100_SXM4|GPU_SKU:V100_PCIE|GPU_SKU:V100S_PCIE|GPU_SKU:V100_SXM2" \
    --mem=60G \
    -o $model.footprinting.log.o \
    -e $model.footprinting.log.e \
    dnalm_bench/batch_scripts/embedding_footprinting/run_embedding_footprinting.sh $model
