library("DESeq2")
library("pheatmap")
library("ggplot2")
library(optparse)

DART_WORK_DIR=Sys.getenv("DART_WORK_DIR")

option_list <- list(
  make_option(c("-c", "--cell_type"), type="character", default="GM12878", help="Cell Type, Options: GM12878, K562, IMR90, HEPG2, H1ESC")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

cell_type = opt$cell_type
print(cell_type)

cts_unrounded_path <- file.path(DART_WORK_DIR, "task_3_peak_classification/input_data/merged_counts_matrix.csv")
coldata_path <- file.path(DART_WORK_DIR, sprintf("task_3_peak_classification/input_data/%s/%s_deseq_input_coldata.csv", cell_type, cell_type))

cts_unrounded <- as.matrix(read.csv(cts_unrounded_path, row.names="peak"))
cts <- round(cts_unrounded)
coldata <- as.matrix(read.csv(coldata_path))

dds <- DESeqDataSetFromMatrix(countData = cts,
                              colData = coldata,
                              design = ~ condition)

featureData <- data.frame(gene=rownames(cts))
mcols(dds) <- DataFrame(mcols(dds), featureData)

dds$condition <- factor(dds$condition)
dds$condition <- relevel(dds$condition, ref = "other")

dds <- DESeq(dds)
res <- results(dds)

resultsNames(dds)

resLFC <- lfcShrink(dds, coef=sprintf("condition_%s_vs_other", cell_type), type="apeglm")

resOrdered <- res[order(res$pvalue),]

ntd <- normTransform(dds)

select <- order(rowMeans(counts(dds,normalized=TRUE)),
                decreasing=TRUE)[1:20]
df <- as.data.frame(colData(dds)[,c("condition","type")])
pheatmap(assay(ntd)[select,], cluster_rows=FALSE, show_rownames=FALSE,
         cluster_cols=FALSE, annotation_col=df)

res_df <- as.data.frame(res)
rownames(res_df) <- rownames(res)

diff_acc_peaks_path <- file.path(DART_WORK_DIR, sprintf("task_3_peak_classification/input_data/%s/diff_acc_peaks.csv", cell_type))

res_df$significant <- ifelse(res_df$padj < 0.001 & abs(res_df$log2FoldChange) > 1, "Significant", "Not significant")
write.csv(res_df, file = diff_acc_peaks_path, row.names = TRUE)
