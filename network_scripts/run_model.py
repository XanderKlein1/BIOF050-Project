import scanpy as sc
import pandas as pd
import numpy as np
from utils import MemoryLogger, ManageQueue
import tensorflow as tf
from tensorflow import keras
from keras.losses import MeanSquaredError
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from network import Encoder, Projector, ContrastiveModel
from train import train
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # non-GUI backend
import matplotlib.pyplot as plt

#Functions:
#Generate a random mask for a given cells x genes matrix
def mask_genes(X, mask_prob=0.2,mask_value=0):
    X_masked = X.copy()
    nnz = X_masked.nnz
    drop = np.random.rand(nnz) < mask_prob
    X_masked.data[drop] = mask_value
    X_masked.eliminate_zeros()
    return X_masked



#Set the root directory
HERE = Path(__file__).resolve().parent.parent.parent
#Import the data from local directory and setup the adata object
adata = sc.read_mtx(HERE / "training_data" / "counts.mtx")
genes = pd.read_csv(HERE / "training_data" / "genes.csv", header=None)[0]
cells = pd.read_csv(HERE / "training_data" / "cells.csv", header=None)[0]
adata.obs_names = cells.astype(str)
adata.var_names = genes.astype(str)

# --- Compute library depth for each cell and add to adata ---
#library_depth = np.sum(adata.X, axis=1)
#adata.obs["library_depth"] = library_depth
#Normalize library depth across the data and store into adata
#scale_factor = (1.0 / adata.obs["library_depth"]).to_numpy()
#adata.layers["X_norm"] = adata.X.tocsr().multiply(scale_factor[:, None])
#log_memory("after adding library depth ")

#--- Augmentation: Random gene masking ---
view1 = mask_genes(adata.X, 0.2, 0)
view2 = mask_genes(adata.X, 0.2, 0)

#--- Build the network architecture ---
encoder = Encoder(adata.n_vars)
projector = Projector()
model = ContrastiveModel(encoder, projector)
model.compile(optimizer='adam')

#--- Convert input data to tensors ---
view1 = tf.convert_to_tensor(view1.toarray(), dtype=tf.float32)
view2 = tf.convert_to_tensor(view2.toarray(), dtype=tf.float32)

#--- Generate training and validation subsets ---
split_idx = int(0.8 * len(view1))
#Create the training split and partition into batches
train_data = tf.data.Dataset.from_tensor_slices((view1[:split_idx], view2[:split_idx]))
train_data = train_data.batch(256).shuffle(1000)
#Create the validation split and partition into batches
val_data = tf.data.Dataset.from_tensor_slices((view1[split_idx:], view2[split_idx:]))
val_data = val_data.batch(256)



tf.config.run_functions_eagerly(True)
#--- Fit the model ---
history = model.fit(train_data, epochs=3, 
          validation_data=val_data,
          callbacks=[ManageQueue()]
          )

#-- Plot the training curve ---
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Curve")
plt.legend()

plt.savefig('training_curve_2.pdf')
plt.close()

#--- Test Top-k recall accuracy ---
#Generate two new masked views of the data
view3 = mask_genes(adata.X, 0.2, 0)
view3 = tf.convert_to_tensor(view3.toarray(), dtype=tf.float32)
view4 = mask_genes(adata.X, 0.2, 0)
view4 = tf.convert_to_tensor(view4.toarray(), dtype=tf.float32)
#Encode both views
z1=encoder(view3, training=False)
z2=encoder(view4, training=False)
#Normalize both views so the dot product becomes cosine similarity
z1 = tf.math.l2_normalize(z1, axis=1)
z2 = tf.math.l2_normalize(z2, axis=1)
#Compute the similarity matrix, where S(i,j) is the cosine similarity between
#cell i from view 1 and cell j from view 2
logits = tf.matmul(z1,z2, transpose_b=True)

#Compute the top-k recall (how frequently a point was within the k nearest
#neighbors of its positive pair)
topk = 0
for i in range(len(z1)):
    best_matches = tf.math.top_k(logits[i], k=5).indices
    if i in best_matches.numpy():
        topk += 1
topk_acc = topk / len(z1)
print(topk_acc)






#Convert the sparse matrix to a dense one (otherwise it won't work with MSE)
#X_dense = adata.X.toarray().astype("float32")
#adata.X = X_dense

#autoencoder = SimpleAutoencoder(adata.n_vars, MeanSquaredError())
#autoencoder.build()

#loss = train(adata, autoencoder, epochs=30, 
#            optimizer=keras.optimizers.Adam(learning_rate=3e-4), batch_size=64)

#Generate the output data from the trained model:
#inputs = {'feature counts': adata.X, 'library depth': adata.obs.library_depth}
#pred = autoencoder.model.predict(inputs, batch_size=64)

