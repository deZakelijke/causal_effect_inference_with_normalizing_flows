import tensorflow as tf
from tensorflow.keras import layers, Model

class FC_net(Model):
    """ Simple fully connected net with not activation after the last layer. """
    def __init__(self, in_dims, out_dims, nr_hidden=1, hidden_dims=256, activation='relu'):
        super.__init__()

        if nr_hidden < 1:
            print("Must have at leas one hidden layer")
            return None

        nn_layers = [layers.Dense(hidden_dims), activation=activation)]

        for i in range(nr_hidden - 1):
            new_layer = layers.Dense(hidden_dims), activation=activation)
            nn_layers.append(new_layer)
 
        nn_layers.append(nn.Linear(out_dims, out_dims))
        self.nn_layers = tf.keras.Sequential(nn_layers)

    def call(self, x):
        return self.nn_layers(x)
