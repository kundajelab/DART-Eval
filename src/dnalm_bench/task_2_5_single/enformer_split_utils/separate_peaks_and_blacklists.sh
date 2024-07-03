#!/bin/sh

# rawpeaks=/oak/stanford/groups/akundaje/projects/chromatin-atlas-2022/DNASE/ENCSR000EMT/peak_calling/ENCSR000EMT_peaks.narrowPeak
# enformerdir=/oak/stanford/groups/akundaje/projects/dnalm_benchmark/enformer_splits_files/enformer_splits/
# outdir=/srv/scratch/patelas/chrombpnet_enformer/GM12878/

rawpeaks=$1
enformerdir=$2
outdir=$3

bedtools intersect -u -f 1.0 -a $rawpeaks -b $enformerdir/sequences_human_train.bed > $outdir/peaks_train.bed
bedtools intersect -u -f 1.0 -a $rawpeaks -b $enformerdir/sequences_human_valid.bed > $outdir/peaks_valid.bed
bedtools intersect -u -f 1.0 -a $rawpeaks -b $enformerdir/sequences_human_test.bed > $outdir/peaks_test.bed

bedtools intersect -u -f 1.0 -a $rawpeaks -b $enformerdir/sequences_human_traintest.bed > $outdir/peaks_traintest.bed
bedtools intersect -u -f 1.0 -a $rawpeaks -b $enformerdir/sequences_human_validtest.bed > $outdir/peaks_validtest.bed
bedtools intersect -u -f 1.0 -a $rawpeaks -b $enformerdir/sequences_human_trainvalid.bed > $outdir/peaks_trainvalid.bed

