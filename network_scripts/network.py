import tensorflow as tf
from keras import layers, models
from keras.layers import BatchNormalization, Dense, Input, Lambda, ReLU
from keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from keras.metrics import Mean
from keras.models import Model
from tensorflow import keras


class SimpleAutoencoder:
    # Instantiate the keras model and define layers
    def __init__(self, num_features: int, loss):
        self.num_features: int = num_features
        self.model = None

        self.enc_dense1 = Dense(1024)
        self.enc_norm1 = BatchNormalization()
        self.enc_relu1 = ReLU()
        self.enc_dense2 = Dense(128)
        self.enc_norm2 = BatchNormalization()
        self.enc_relu2 = ReLU()

        # Latent space will have 32 dimensions
        self.latent = Dense(32, name="latent")

        # Projection head
        self.proj_dense1 = Dense(64)
        self.proj_norm = BatchNormalization()
        self.proj_relu = ReLU()
        self.proj_dense2 = Dense(32)

        self.loss = loss

    # Build the layer architecture
    def build(self):
        # Create a tensor for the features input layer
        self.input_layer = Input(shape=(self.num_features,), name="feature counts")

        # Create a tensor for the library depth input layer
        self.ld_layer = Input(shape=(1,), name="library depth")

        # Encoder layers:
        x = self.enc_dense1(self.input_layer)
        x = self.enc_norm1(x)
        x = self.enc_relu1(x)

        x = self.enc_dense2(x)
        x = self.enc_norm2(x)
        x = self.enc_relu2(x)

        # Latent space:
        z = self.latent(x)
        self.center = z

        # Projection head:
        p = self.proj_dense1(z)
        p = self.proj_norm(p)
        p = self.proj_relu(p)

        self.output = self.proj_dense2(p)

        # self.decoder_output = x
        # self.scale_output()

    # Pass the decoded layer through another layer to return to # of input features
    # Then scale each bin by its library depth
    def scale_output(self):
        mean = Dense(self.num_features, name="mean")(self.decoder_output)

        # Mean is now a (cells, features) array for one batch

        # We pass through another layer that multiplies each row of mean by the size factor for that cell. This essentially scales the gene vector of each cell in the batch by the library depth of that cell.
        output = Lambda(lambda l: l[0] * tf.reshape(l[1], (-1, 1)))(
            [mean, self.ld_layer]
        )
        self.model = Model(inputs=[self.input_layer, self.ld_layer], outputs=output)
        self.encoder = self.get_encoder()

    def get_encoder(self):
        ret = Model(
            inputs=self.model.input, outputs=self.model.get_layer("latent").output
        )
        return ret


### --- ###


### --- ###
class Encoder(Model):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features: int = num_features

        self.enc_dense1 = Dense(1024)
        self.enc_norm1 = BatchNormalization()
        self.enc_relu1 = ReLU()
        self.enc_dense2 = Dense(128)
        self.enc_norm2 = BatchNormalization()
        self.enc_relu2 = ReLU()

        self.latent = Dense(32, name="latent")

    def call(self, x, training=False):

        x = self.enc_dense1(x)
        x = self.enc_norm1(x, training=training)
        x = self.enc_relu1(x)

        x = self.enc_dense2(x)
        x = self.enc_norm2(x, training=training)
        x = self.enc_relu2(x)

        z = self.latent(x)
        return z


### --- ###
class Projector(Model):
    def __init__(self):
        super().__init__()

        self.proj_dense1 = Dense(64)
        self.proj_norm = BatchNormalization()
        self.proj_relu = ReLU()
        self.proj_dense2 = Dense(32)

    def call(self, z, training=False):
        p = self.proj_dense1(z)
        p = self.proj_norm(p, training=training)
        p = self.proj_relu(p)
        p = self.proj_dense2(p)

        return p


### --- ###
class ContrastiveModel(Model):
    def __init__(
        self, encoder, projector, temperature=0.2, queue_size=1280, feat_dim=32
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.temperature = temperature
        self.loss_tracker = Mean(name="loss")
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        # Define the memory bank
        self.queue_size = queue_size
        self.feat_dim = feat_dim

        self.queue = tf.Variable(
            tf.zeros([queue_size, feat_dim]), trainable=False, dtype=tf.float32
        )

    @property
    def metrics(self):
        return [self.loss_tracker]

    # Attach the current batch to the queue and pop off the oldest batch.
    def update_queue(self, queue, new_obs):
        batch_size = tf.shape(new_obs)[0]
        queue_size = tf.shape(queue)[0]

        queue_tail = queue[: queue_size - batch_size]
        queue = tf.concat([new_obs, queue_tail], axis=0)

        return queue

    def call(self, inputs, training=False):
        views = inputs
        z_list, p_list = [], []
        for x in views:
            z = self.encoder(x, training=training)
            p = self.projector(z, training=training)
            z_list.append(z)
            p_list.append(p)
        return z_list, p_list

    # Implementation of the InfoNCE loss function
    def contrastive_loss(self, p_list):
        # Normalize the input data so the dot product becomes cosine similarity
        p_list_norm = []
        for p in p_list:
            p = tf.math.l2_normalize(p, axis=1)
            p_list_norm.append(p)

        # Generate the logits matrix. Each row specifies the similarity score between cell i and all other
        # cells from the batch. logits[i,i] represents the positive pair, while logits[i,j] for j != i represent
        # the negative pairs
        logits = (
            tf.matmul(p_list_norm[0], p_list_norm[1], transpose_b=True)
            / self.temperature
        )
        labels = tf.range(tf.shape(p_list_norm[0])[0])

        # Compute the loss function for each view of the data.
        loss_1 = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )

        loss_2 = tf.keras.losses.sparse_categorical_crossentropy(
            labels, tf.transpose(logits), from_logits=True
        )

        return tf.reduce_mean(loss_1 + loss_2) / 2.0

    # Core training loop for the contrastive model
    def train_step(self, data):
        views = data
        # Forward pass, recording operations with GradientTape
        with tf.GradientTape() as tape:
            z_list, p_list = self(views, training=True)
            loss = self.contrastive_loss(p_list)

        # Compute gradients
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # (temporary location)
        self.queue = self.update_queue(self.queue, p_list[0])

        self.loss_tracker.reset_state()
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    # Core test loop for the contrastive model
    def test_step(self, data):
        views = data
        z_list, p_list = self(views, training=False)
        loss = self.contrastive_loss(p_list)

        breakpoint()
        self.queue = self.update_queue(self.queue, p_list[0])

        self.loss_tracker.reset_state()
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
