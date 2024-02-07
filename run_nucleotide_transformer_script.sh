#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH -G 1
#SBATCH --mem=500GB
#SBATCH --partition=akundaje

eval "$(conda shell.bash hook)"
conda activate dnalm
export HF_HOME=/oak/stanford/groups/akundaje/arpitas/chrombench/cache

python3 run_nucleotide_transformer.py "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT" 0 "nucleotide-transformer-500m-human-ref"
