import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.losses import MeanSquaredError
from keras.layers import Input, Dense, Lambda, BatchNormalization, ReLU
from keras import models
from keras.models import Model

class SimpleAutoencoder():
    #Instantiate the keras model and define layers
    def __init__(self, num_features):
        self.num_features = num_features
        self.model = None

        self.enc_dense1 = Dense(1024)
        self.enc_norm1 = BatchNormalization()
        self.enc_relu1 = ReLU()
        self.enc_dense2 = Dense(128)
        self.enc_norm2 = BatchNormalization()
        self.enc_relu2 = ReLU()

        #Latent space will have 32 dimensions
        self.latent = Dense(32, name='latent')

        self.dec_dense1 = Dense(128)
        self.dec_norm1 = BatchNormalization()
        self.dec_relu1 = ReLU()
        self.dec_dense2 = Dense(1024)
        self.dec_norm2 = BatchNormalization()
        self.dec_relu2 = ReLU()

        self.loss = MeanSquaredError()
        
    #Build the layer architecture
    def build(self):
        #Create a tensor for the features input layer
        self.input_layer = Input(shape=(self.num_features,), name="feature counts")

        #Create a tensor for the library depth input layer
        self.ld_layer = Input(shape=(1,), name="library depth")
        #note: we can add a dropout layer here later

        #Encoder layers:
        x = self.enc_dense1(self.input_layer)
        x = self.enc_norm1(x)
        x = self.enc_relu1(x)
     
        x = self.enc_dense2(x)
        x = self.enc_norm2(x)
        x = self.enc_relu2(x)

        #Latent space:
        z = self.latent(x)
        self.center = z

        #Decoder layers:
        x = self.dec_dense1(z)
        x = self.dec_norm1(x)
        x = self.dec_relu1(x)
     
        x = self.dec_dense2(x)
        x = self.dec_norm2(x)
        x = self.dec_relu2(x)

        self.decoder_output = x
        self.scale_output()

    #Pass the decoded layer through another layer to return to # of input features
    #Then scale each bin by its library depth
    def scale_output(self):
        mean = Dense(self.num_features, name='mean')(self.decoder_output)

        #Mean is now a (cells, features) array for one batch

        #We pass through another layer that multiplies each row of mean by the size factor for that cell. This essentially scales the gene vector of each cell in the batch by the library depth of that cell.
        output = Lambda(lambda l: l[0]*tf.reshape(l[1], (-1,1)))([mean, self.ld_layer])
        self.model = Model(inputs=[self.input_layer,self.ld_layer], outputs=output)
        self.encoder = self.get_encoder()

    def get_encoder(self):
        #Note: self.center tensor is probably NOT scaled yet (since we never ran scale_output on it)
        ret = Model(inputs=self.model.input, outputs=self.model.get_layer('latent').output)
        return ret
