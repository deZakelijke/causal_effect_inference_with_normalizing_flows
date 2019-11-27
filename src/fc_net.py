import tensorflow as tf
from tensorflow.keras import layers, Model

class FC_net(Model):
    """ Simple fully connected net with not activation after the last layer. """
    def __init__(self, in_dims, out_dims, name_tag, nr_hidden=2, hidden_size=256, activation='elu', debug=False):
        super().__init__()

        self.debug = debug
        self.name_tag = name_tag

        assert nr_hidden >= 1, "Must have at leas one hidden layer"
        assert name_tag != "", "Name tag can't be an empty stirng"


        with tf.name_scope(f"FC/{self.name_tag}") as scope:
            new_layer = layers.Dense(hidden_size, activation=activation, dtype="float64", name="dense_0")
            new_layer.build((None, in_dims))
            nn_layers = [new_layer]

            i = 0
            for i in range(nr_hidden - 1):
                new_layer = layers.Dense(hidden_size, activation=activation, dtype="float64", name=f"dense_{i + 1}")
                new_layer.build((None, hidden_size))
                nn_layers.append(new_layer)
 
            new_layer = layers.Dense(out_dims, activation=None, dtype="float64", name=f"dense_{i + 2}")
            new_layer.build((None, hidden_size))
            nn_layers.append(new_layer)

            self.nn_layers = nn_layers
            #self.nn_layers = tf.keras.Sequential(nn_layers)
            #self.nn_layers.build((None, in_dims))


    @tf.function
    def call(self, x, step):
        with tf.name_scope(f"FC/{self.name_tag}/FC") as scope:
            for i in range(len(self.nn_layers)):
                x = self.nn_layers[i](x)
                name = self.nn_layers[i].name
                tf.summary.histogram(f"{name}/weight", self.nn_layers[i].weights[0], step=step)
                tf.summary.histogram(f"{name}/bias", self.nn_layers[i].weights[1], step=step)
            return x
