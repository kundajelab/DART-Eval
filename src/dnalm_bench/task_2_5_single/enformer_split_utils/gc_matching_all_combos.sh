#!/bin/sh

filedir=/srv/scratch/patelas/chrombpnet_enformer/GM12878/
bigwigfile=/oak/stanford/groups/akundaje/projects/chromatin-atlas-2022/DNASE/ENCSR000EMT/preprocessing/bigWigs/ENCSR000EMT.bigWig
biasmodel=/oak/stanford/groups/akundaje/projects/chromatin-atlas-2022/reference/HEPG2_DNASE_PE/fold_2/bias.h5
taskname=chrombpnet_enformer_splits
mkdir $filedir/$taskname/
cd /users/patelas/lib/chrombpnet-soft-multitask/

python /users/patelas/chrombench/src/dnalm_bench/task_2_5_single/enformer_split_utils/run_gc_matching.py --peak_file $filedir/peaks_train.bed --reference_genome /oak/stanford/groups/akundaje/patelas/reference/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta --chrom_size_file /oak/stanford/groups/akundaje/patelas/reference/hg38/hg38.chrom.sizes --blacklist $filedir/peaks_validtest.bed --output_dir $filedir/train_negs/

python /users/patelas/chrombench/src/dnalm_bench/task_2_5_single/enformer_split_utils/run_gc_matching.py --peak_file $filedir/peaks_valid.bed --reference_genome /oak/stanford/groups/akundaje/patelas/reference/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta --chrom_size_file /oak/stanford/groups/akundaje/patelas/reference/hg38/hg38.chrom.sizes --blacklist $filedir/peaks_traintest.bed --output_dir $filedir/valid_negs/

python /users/patelas/chrombench/src/dnalm_bench/task_2_5_single/enformer_split_utils/run_gc_matching.py --peak_file $filedir/peaks_test.bed --reference_genome /oak/stanford/groups/akundaje/patelas/reference/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta --chrom_size_file /oak/stanford/groups/akundaje/patelas/reference/hg38/hg38.chrom.sizes --blacklist $filedir/peaks_trainvalid.bed --output_dir $filedir/test_negs/

cat $filedir/peaks_train.bed $filedir/peaks_valid.bed $filedir/peaks_test.bed > $filedir/all_peaks_split_ordered.bed

cat $filedir/train_negs/negatives_with_summit.bed $filedir/valid_negs/negatives_with_summit.bed $filedir/test_negs/negatives_with_summit.bed > $filedir/all_negatives_split_ordered.bed

python /users/patelas/chrombench/src/dnalm_bench/task_2_5_single/enformer_split_utils/make_fold_json.py $filedir/peaks_train.bed $filedir/peaks_valid.bed $filedir/peaks_test.bed $filedir/$taskname/fold_3.json

mkdir $filedir/file_lists/

echo $filedir/all_peaks_split_ordered.bed > $filedir/file_lists/peak_files.txt
echo $filedir/all_negatives_split_ordered.bed > $filedir/file_lists/nonpeak_files.txt
echo $taskname > $filedir/file_lists/task_list.txt
echo $bigwigfile > $filedir/file_lists/bigwig_files.txt
echo /oak/stanford/groups/akundaje/patelas/reference/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta > $filedir/file_lists/genome_files.txt

bash run_chrombpnet_training.sh $filedir/file_lists/task_list.txt $filedir/file_lists/genome_files.txt $filedir/file_lists/bigwig_files.txt $filedir/file_lists/peak_files.txt $filedir/file_lists/nonpeak_files.txt $filedir 3 $biasmodel $filedir DNASE_PE 