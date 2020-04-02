import tensorflow as tf
from tensorflow.keras import layers, Model


class CNN(Model):
    """ Convolutional Neural Network."""

    def __init__(self, in_dims, out_dims, name_tag, nr_convs=3, nr_hidden=1,
                 activation='elu', filters=32, skip_connection=True,
                 debug=False):
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

        if skip_connection:
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

    def __init__(self):
        super().__init__()

    @tf.function
    def call(self, x, step, training=False):
        pass
