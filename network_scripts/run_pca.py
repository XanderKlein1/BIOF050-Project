import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.decomposition import PCA

def run_pca(adata):

    #Normalize library depth across cells
    X = adata.X
    library_depth = X.sum(axis=1, keepdims=True)
    avg_lib_depth = library_depth.mean()
    X_norm = X / library_depth * avg_lib_depth

    #Log transform the data
    X_norm = np.log1p(X_norm)

    #Gene-wise scaling (set mean = 0 and sd = 1 for all genes)
    X_norm = (X_norm - X_norm.mean(axis=0)) / X_norm.std(axis=0)

    #Run PCA on the data
    pca = PCA(n_components=32)
    X_pca = pca.fit_transform(X_norm)