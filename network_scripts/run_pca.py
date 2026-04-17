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

    #Scale genes