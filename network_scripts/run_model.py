import scanpy as sc
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from network import SimpleAutoencoder
from train import train


#Import the data from local directory and setup the adata object
adata = sc.read_mtx("counts.mtx")
print(adata.X)
genes = pd.read_csv("genes.csv", header=None)[0]
cells = pd.read_csv("cells.csv", header=None)[0]
adata.obs_names = cells.astype(str)
adata.var_names = genes.astype(str)

#Convert the sparse matrix to a dense one (otherwise it won't work with MSE)
X_dense = adata.X.toarray().astype("float32")
adata.X = X_dense
#Compute library depth for each cell and add to adata
library_depth = np.sum(adata.X, axis=1)
adata.obs["library_depth"] = library_depth

autoencoder = SimpleAutoencoder(adata.n_vars)
autoencoder.build()

loss = train(adata, autoencoder, epochs=10, optimizer=keras.optimizers.Adam(learning_rate=3e-4))

