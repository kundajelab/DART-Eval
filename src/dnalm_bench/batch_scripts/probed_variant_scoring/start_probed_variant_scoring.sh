#!/bin/sh

model=$1
variantsbed=$2
countstsv=$3
genome=$4
celltype=$5

export HF_HOME=/scratch/groups/akundaje/dnalm_benchmark/.cache

sbatch --export=ALL --requeue \
    -J variantscoring.$model.$celltype \
    -p akundaje,gpu,owners -t 24:00:00 \
    -G 1 -C "GPU_MEM:80GB|GPU_SKU:A100_PCIE|GPU_SKU:A100_SXM4|GPU_SKU:V100_PCIE|GPU_SKU:V100S_PCIE|GPU_SKU:V100_SXM2" \
    --mem=60G \
    -o $model.$celltype.vscoring.log.o \
    -e $model.$celltype.vscoring.log.e \
    dnalm_bench/batch_scripts/probed_variant_scoring/run_probed_variant_scoring.sh $model $variantsbed $countstsv $genome $celltype
