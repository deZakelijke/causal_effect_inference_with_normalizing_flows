import tensorflow as tf
from tensorflow.keras import layers, Model


class FC_net(Model):
    """ Simple fully connected net with not activation after the last layer."""

    def __init__(self, in_dims, out_dims, name_tag, nr_hidden=2,
                 hidden_size=256, activation='elu', squeeze=False,
                 squeeze_dims=None, debug=False):
        """
        Parameters
        ----------
        in_dims : int
            The number of nodes for the input layer.
        out_dims : int
            Thenumber of nodes for the output layer.
        name_tag : str
            A tag used to identify the model in tensorboard. Should be unique
            for all instances of the class.
        nr_hidden : int
            The number of hidden layers.
        hidden_size : int
            The number of nodes in the hidden layers.
        activation : str
            Activation function after every layer.
        debug : bool
            Flag to enable debug mode. Currently unused.
        """

        super().__init__()

        self.debug = debug
        self.name_tag = name_tag

        assert nr_hidden >= 1 and type(nr_hidden) == int,\
            "Must have at leas one hidden layer"
        assert name_tag != "", "Name tag can't be an empty stirng"

        if squeeze:
            out_dims = squeeze_dims

        with tf.name_scope(f"FC/{self.name_tag}") as scope:
            new_layer = layers.Dense(hidden_size, activation=activation,
                                     dtype="float64", name="dense_0")
            new_layer.build((None, in_dims))
            nn_layers = [new_layer]

            i = 0
            for i in range(nr_hidden - 1):
                new_layer = layers.Dense(hidden_size, activation=activation,
                                         dtype="float64",
                                         name=f"dense_{i + 1}")
                new_layer.build((None, hidden_size))
                nn_layers.append(new_layer)

            new_layer = layers.Dense(out_dims, activation=None,
                                     dtype="float64", name=f"dense_{i + 2}")
            new_layer.build((None, hidden_size))
            nn_layers.append(new_layer)

            self.nn_layers = nn_layers

    @tf.function
    def call(self, x, step, training=False):
        with tf.name_scope(f"FC/{self.name_tag}/FC") as scope:
            for i in range(len(self.nn_layers)):
                x = self.nn_layers[i](x)
                name = self.nn_layers[i].name
                if training and step is not None:
                    tf.summary.histogram(f"{name}/weight",
                                         self.nn_layers[i].weights[0],
                                         step=step)
                    tf.summary.histogram(f"{name}/bias",
                                         self.nn_layers[i].weights[1],
                                         step=step)
            return x
