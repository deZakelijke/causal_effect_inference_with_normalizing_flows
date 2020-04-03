import tensorflow as tf
from tensorflow.keras import layers, Model, activations


class ResNet(Model):
    """ Convolutional Neural Network."""

    def __init__(self, in_dims, out_dims, name_tag, nr_convs=3, nr_hidden=1,
                 activation='elu', filters=32, residual_connection=True,
                 debug=False):
        """
        Parameters
        ----------
        in_dims : (int, int, int)
            The shape of the input of the network. Channels should be last.

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

        filters : int
            Number of filters to use in the first convolutional block.

        skip_connection : bool
            Flag to use residual convolutional blocks instead of regular
            convolutional blocks.

        debug : bool
            Flag to enable debug mode. Currently unused.

        """

        super().__init__()

        self.debug = debug
        self.name_tage = name_tag
        assert name_tag != "", "Name tag can't be an empty stirng"

        if residual_connection:
            block = ResidualConvBlock
        else:
            block = ConvBlock

        with tf.name_scope(f"CNN/{name_tag}") as scope:
            new_block = block()

    @tf.function
    def call(self, x, step, training=False):
        pass


class ConvBlock(Model):
    """ Single block of a convolutional neural network.

    Contains a convolution with nonlinearity, maxpooling and batchnorm.
    """

    def __init__(self):
        super().__init__()

    @tf.function
    def call(selc, x, step, training=False):
        pass


class ResidualConvBlock(Model):
    """ Single block of a convolutional neural network with a skip connection.

    Contains a convolution with nonlinearity, maxpooling, batch norm and
    a skip connection.
    """

    def __init__(self, dims, name_tag, activation, filters):
        """
        Parameters
        ----------
        dims : (int, int, int, int)
            The shape of the input and output of block.
            Channels should be last.

        name_tag : str
            Tag to identify the block in Tensorboard.

        activation : str
            Activation function used after batch normalisation.

        filters : int
            Number of filters in the convolutional block.
        """

        super().__init__(dtype=tf.float64)
        self.name_tag = name_tag
        self.activation = eval(f"activations.{activation}")
        BN_1 = layers.BatchNormalization(axis=3, dtype=tf.float64,
                                         fused=False, name="BN_1")
        BN_1.build(dims)
        Conv_1 = layers.Conv2D(filters, kernel_size=(1, 1),
                               padding="same",
                               data_format="channels_last",
                               activation=None, use_bias=False,
                               dtype=tf.float64, name="Conv_1")
        Conv_1.build(dims)
        BN_2 = layers.BatchNormalization(axis=3, dtype=tf.float64,
                                         fused=False, name="BN_2")
        BN_2.build(dims)
        Conv_2 = layers.Conv2D(filters, kernel_size=(3, 3),
                               padding='same',
                               data_format="channels_last",
                               activation=None, use_bias=False,
                               dtype=tf.float64, name="Conv_2")
        Conv_2.build(dims)
        BN_3 = layers.BatchNormalization(axis=3, dtype=tf.float64,
                                         fused=False, name="BN_3")
        BN_3.build(dims)
        Conv_3 = layers.Conv2D(filters, kernel_size=(1, 1),
                               padding="same",
                               data_format="channels_last",
                               activation=None, use_bias=True,
                               dtype=tf.float64, name="Conv_3")
        Conv_3.build(dims)
        
        self.nn_layers = [BN_1, Conv_1, BN_2, Conv_2, BN_3, Conv_3]

    @tf.function
    def call(self, x, step, training=False):
        h = x
        with tf.name_scope(f"ResBlock/{self.name_tag}") as scope:
            for i in range(0, 6, 2):
                h = self.nn_layers[i](h, training=training)
                h = self.activation(h)
                h = self.nn_layers[i + 1](h, training=training)
            if training and step is not None:
                self.log_weights(step)
        return x + h

    def log_weights(self, step):
        for i in range(0, 6, 2):
            name = self.nn_layers[i].name
            weights = self.nn_layers[i].weights
            tf.summary.histogram(f"{name}/gamma", weights[0], step=step)
            tf.summary.histogram(f"{name}/beta", weights[1], step=step)
            name = self.nn_layers[i + 1].name
            weights = self.nn_layers[i + 1].weights
            tf.summary.histogram(f"{name}/kernel", weights[0], step=step)
        tf.summary.histogram(f"{name}/bias", weights[1], step=step)


def test_residual_block():
    """ Unit test of the residual block."""
    dims = (None, 10, 20, 32)
    name_tag = "test"
    activation = "relu"
    filters = 32
    x = tf.zeros((2, 10, 20, 32), dtype=tf.float64)

    block = ResidualConvBlock(dims, name_tag, activation, filters)
    block(x, 0, training=True)
    print(block.summary())

if __name__ == "__main__":
    test_residual_block()
