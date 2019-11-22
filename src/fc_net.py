import tensorflow as tf
from tensorflow.keras import layers, Model

class FC_net(Model):
    """ Simple fully connected net with not activation after the last layer. """
    def __init__(self, out_dims, nr_hidden=3, hidden_size=256, activation='relu'):
        # TODO add input dim and build nn at the end of init. Allows summary.
        super().__init__()

        if nr_hidden < 1:
            print("Must have at leas one hidden layer")
            return None

        nn_layers = [layers.Dense(hidden_size, activation=activation, dtype="float64")]
        for i in range(nr_hidden - 1):
            new_layer = layers.Dense(hidden_size, activation=activation, dtype="float64")
            nn_layers.append(new_layer)
 
        nn_layers.append(layers.Dense(out_dims, activation=None, dtype="float64"))

        self.nn_layers = tf.keras.Sequential(nn_layers)

    @tf.function
    def call(self, x):
        for i in range(len(self.nn_layers.weights)):
            tf.summary("weight", self.nn_layers[i].weights[0])
            tf.summary("bias", self.nn_layers[i].weights[1])


        return self.nn_layers(x)
