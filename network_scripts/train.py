from network import SimpleAutoencoder

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential


def train(adata, network, epochs=1, batch_size=32, output_dir=None, validation_split=0.1, 
          optimizer=keras.optimizers.Adam(learning_rate=1e-3)):
    
    model = network.model
    model.compile(optimizer = optimizer, loss = network.loss)

    #Create some callbacks to regulate the model while it runs:
    callbacks = []

    #Save a snapshot (checkpoint) of the network's weights whenever a new "best" model is found
    if output_dir is not None:
        checkpointer_cb = ModelCheckpoint(filepath="%s/weights.h5" % output_dir, 
                                       save_weights_only=True, save_best_only=True)
        callbacks.append(checkpointer_cb)
    
    #Stop the training early if validation loss doesn't improve after 2 consecutive epochs.
    earlystop_cb = EarlyStopping(monitor='val_loss',patience=4, restore_best_weights=True)
    callbacks.append(earlystop_cb)

    #Set inputs + target output for the model and train:
    #note: how to format the input data?

    inputs = {'feature counts': adata.X, 'library depth': adata.obs.library_depth}
    output = adata.X

    
    #Now we can fit the model and compute loss
    loss = model.fit(inputs, output, batch_size=batch_size, epochs=epochs, 
                     callbacks=callbacks, validation_split=validation_split)
    
    return loss