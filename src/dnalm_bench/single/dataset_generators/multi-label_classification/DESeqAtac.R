library("DESeq2")
library("pheatmap")
library("ggplot2")
library(optparse)


option_list <- list(
  make_option(c("-c", "--cell_type"), type="character", default="GM12878", help="Cell Type, Options: GM12878, K562, IMR90, HEPG2, H1ESC")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

cell_type = opt$cell_type
print(cell_type)

cts_unrounded <- as.matrix(read.csv("/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/merged_counts_matrix.csv", row.names="peak"))
cts <- round(cts_unrounded)
coldata <- as.matrix(read.csv(sprintf("/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/%s/%s_deseq_input_coldata.csv", cell_type, cell_type)))

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

png(sprintf("/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/%s/MA_plot.png", cell_type))
plotMA(resLFC, ylim=c(-2,2))
dev.off()

png(sprintf("/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/%s/min_res_padj.png", cell_type))
plotCounts(dds, gene=which.min(res$padj), intgroup="condition")
dev.off()

write.csv(as.data.frame(resOrdered), file=sprintf("/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/%s/condition_treated_results.csv", cell_type))

ntd <- normTransform(dds)
png(sprintf("/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/%s/ConditionCountPlot.png", cell_type))
#meanSdPlot(assay(ntd))

select <- order(rowMeans(counts(dds,normalized=TRUE)),
                decreasing=TRUE)[1:20]
df <- as.data.frame(colData(dds)[,c("condition","type")])
pheatmap(assay(ntd)[select,], cluster_rows=FALSE, show_rownames=FALSE,
         cluster_cols=FALSE, annotation_col=df)
dev.off()

png(sprintf("/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/%s/VolcanoPlot.png", cell_type))
res_df <- as.data.frame(res)
rownames(res_df) <- rownames(res)
write.csv(res_df, file = sprintf("/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/%s/diff_acc_peaks.csv", cell_type), row.names = TRUE)

# Creating a significance label

res_df$significant <- ifelse(res_df$padj < 0.001 & abs(res_df$log2FoldChange) > 1, "Significant", "Not significant")

# Enhanced volcano plot
volcano_plot <- ggplot(res_df, aes(x=log2FoldChange, y=-log10(pvalue), color=significant)) +
  geom_point(alpha=0.5) +
  scale_color_manual(values=c("Significant"="red", "Not significant"="black")) +
  theme_minimal() +
  labs(x="Log2 Fold Change", y="-Log10 p-value") +
  theme(axis.text=element_text(face="bold", color="black"),
        title=element_text(face="bold"))

# Display the plot
print(volcano_plot)

write.csv(res_df, file = sprintf("/oak/stanford/groups/akundaje/projects/dnalm_benchmark/cell_line_data/%s/diff_acc_peaks.csv", cell_type), row.names = TRUE)
