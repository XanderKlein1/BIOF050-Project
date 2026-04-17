import scanpy as sc
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from network import SimpleAutoencoder
from train import train
from pathlib import Path
import matplotlib.pyplot as plt


#Set the root directory
HERE = Path(__file__).resolve().parent.parent.parent
#Import the data from local directory and setup the adata object
adata = sc.read_mtx(HERE / "training_data" / "counts.mtx")
genes = pd.read_csv(HERE / "training_data" / "genes.csv", header=None)[0]
cells = pd.read_csv(HERE / "training_data" / "cells.csv", header=None)[0]
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

loss = train(adata, autoencoder, epochs=30, optimizer=keras.optimizers.Adam(learning_rate=3e-4), batch_size=64)

#Plot the training curve
plt.plot(loss.history["loss"], label="training loss")
plt.plot(loss.history["val_loss"], label="validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curve")
plt.show()

#Generate the output data from the trained model:
inputs = {'feature counts': adata.X, 'library depth': adata.obs.library_depth}
pred = autoencoder.model.predict(inputs, batch_size=64)

#We want to evaluate how good the loss value of our model is, so we will compare it to various null models.
#First we will compare to the null model where we set all our outputs to mean(X):
null_mean = adata.X.mean(axis=0)
null_mean_error = ((adata.X - null_mean)**2).mean()

#Next we can compare to the library-depth scaled mean:
scaled_mean = adata.obs.library_depth * null_mean / null_mean.sum()
scaled_mean_error = ((adata.X - scaled_mean)**2).mean()

#Compute the model error:
model_error = ((adata.X - pred)**2).mean()

#Compare thee model error to the null errors
print(null_mean_error)
print(scaled_mean_error)
print(model_error)

