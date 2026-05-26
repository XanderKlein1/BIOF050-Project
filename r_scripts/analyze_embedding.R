library(here)
library(Seurat)
library(Matrix)
library(ggplot2)
library(SeuratDisk)
library(anndata)

# --- Setup the Data --- 

#Set current directory info
here::i_am("repo/r_scripts/analyze_embedding.R")

#Read in the base data as a seurat object
intestine <- Load10X_Spatial(data.dir=here("../data"))

#Read in the embedding data and store it into a matrix object
anndata <- read_h5ad(here("contrastive_embeddings.h5ad"))
embedding_counts <- as.matrix(anndata$obsm$X_latent)
rownames(embedding_counts) <- colnames(LayerData(intestine, "counts"))
colnames(embedding_counts) <- paste0("Feature ", 1:32)

#Attach the embedding to the seurat object.
intestine[["latent"]] <- CreateDimReducObject(
  embeddings = embedding_counts,
  key = "latent_",
  assay = DefaultAssay(intestine)
)

# --- Generate visualizations of the embedding ---

#Build nearest neighbor graph from latent embeddings, and generate clusters
intestine <- FindNeighbors(intestine,reduction = "latent",k.param=80)
intestine <- FindClusters(intestine, resolution = 0.1)

#Project clusters with UMAP and visualize
intestine <- RunUMAP(intestine, reduction = "latent", dims = 1:32)
DimPlot(intestine, reduction = "umap")


# --- Examine the latent space collapse ---

#Convert the embedding of each cell into a string 'signature'
sig <- apply(embedding_counts, 1, function(x) paste(round(x, 6), collapse = "_"))

#Create a contingency table of unique signatures (and counts / signature)
tab <- sort(table(sig), decreasing = TRUE)

#Plot the counts per signature
barplot(tab[1:dim(tab)],
        las = 2,
        main = "Counts at each latent-space point (collapse severity)",
        ylab="Counts",
        xlab="Unique latent points",
        arg.names=NULL,
        axisnames=FALSE)
