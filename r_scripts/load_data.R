#load_data.R
#Loads in the Visium HD dataset that will be used for training the autoencoder.
#Applies basic QC to remove low-count bins
#
#Dataset is pulled from: https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-intestine
#

library(here)
library(Seurat)
library(Matrix)
library(ggplot2)

here::i_am("repo/r_scripts/load_data.R")
localdir <- here("../data")
intestine <- Load10X_Spatial(data.dir = localdir, bin.size = c(8,16))

#Extract counts from the 16um bins data:
counts <- GetAssayData(intestine, assay = "Spatial.016um", layer="counts")
counts <- t(counts)

#Filter to remove lowly expressed bins:
nCounts <- rowSums(counts)
plot(nCounts, ylim=c(0,10000))
plot(nCounts, ylim=c(0,1000))
plot(nCounts, ylim=c(0,100))
#Visually, we can see a cutoff around 10 counts, so this is where we will set our filter point.
keep_cells <- nCounts>10

#Similarly, we will filter out lowly expressed genes:
geneCounts <- colSums(counts)
plot(geneCounts, ylim = c(0, 1000))
plot(geneCounts, ylim = c(0, 100))
plot(geneCounts, ylim = c(0, 30))
#We can see that there appears to be a cutoff in count density at around 4 total counts.
#So we will filter out genes with less than 4 total counts across all bins.
keep_genes <- geneCounts > 4

#Applying both cell and gene filters:
filtered_counts <- counts[keep_cells, keep_genes]

#Store as an AnnData object for downstream analysis in Python
writeMM(filtered_counts, here("training_data", "counts.mtx"))
write.table(colnames(filtered_counts), here("training_data", "genes.csv"), row.names=FALSE, col.names=FALSE)
write.table(rownames(filtered_counts), here("training_data", "cells.csv"), row.names=FALSE, col.names=FALSE)

