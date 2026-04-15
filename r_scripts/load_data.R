#load_data.R
#Loads in the Visium HD dataset that will be used for training the autoencoder.
#Applies basic QC to remove low-count bins
#
#Dataset is pulled from: https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-intestine
#

library(here)
library(Seurat)
library(Matrix)

here::i_am("repo/r_scripts/load_data.R")
localdir <- here("../data")
intestine <- Load10X_Spatial(data.dir = localdir, bin.size = c(8,16))

#Extract counts from the 16um bins data:
counts <- GetAssayData(intestine, assay = "Spatial.016um", layer="counts")
counts <- t(counts)

#Store as an AnnData object for downstream analysis in Python
writeMM(counts, here("training_data", "counts.mtx"))
write.table(colnames(counts), here("training_data", "genes.csv"), row.names=FALSE, col.names=FALSE)
write.table(rownames(counts), here("training_data", "cells.csv"), row.names=FALSE, col.names=FALSE)

