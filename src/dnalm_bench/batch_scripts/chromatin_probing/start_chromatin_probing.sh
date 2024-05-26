#!/bin/sh

model=$1
celltype=$2

sbatch --export=ALL --requeue \
    -J $model.$celltype \
    -p akundaje,gpu,owners -t 48:00:00 \
    -G 1 -C "GPU_MEM:80GB|GPU_MEM:40GB|GPU_MEM:32GB|GPU_MEM:24GB|GPU_MEM:16GB|GPU_SKU:A100_PCIE|GPU_SKU:A100_SXM4|GPU_SKU:V100_PCIE|GPU_SKU:TITAN_V|GPU_SKU:V100S_PCIE|GPU_SKU:V100_SXM2" \
    --mem=60G \
    -o $model.$celltype.probing.log.o \
    -e $model.$celltype.probing.log.e \
    batch_scripts/chromatin_probing/run_chromatin_probing.sh $model $celltype