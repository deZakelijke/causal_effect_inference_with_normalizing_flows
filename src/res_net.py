import tensorflow as tf
from tensorflow.keras import Model, activations
from tensorflow.keras.layers import BatchNormalization, Conv2D


class ResNet(Model):
    """Residual Convolutional Neural Network."""

    def __init__(self, in_dims, out_dims, name_tag, nr_res_blocks=3,
                 activation='elu', filters=32, skip_connection=True,
                 debug=False):
        """
        Parameters
        ----------
        in_dims : (int, int, int)
            The shape of the input of the network. Channels should be last.

        out_dims : (int, int, int)
            The shape of the input of the network. Channels should be last.

        name_tag : str
            A tag used to identify the model in tensorboard. Should be unique
            for all instances of the class.

        nr_res_blocks : int
            The number of residual blocks in the middle of the networks.
            Each residual block has filter number of filters.

        activation : str
            Activation function after every layer.

        filters : int
            Number of filters to use in the residual blocks.

        skip_connection : bool
            Flag to use skip connections inbetween the residual blocks.

        debug : bool
            Flag to enable debug mode. Currently unused.
        """

        super().__init__()

        self.debug = debug
        self.name_tag = name_tag
        self.activation = eval(f"activations.{activation}")
        assert name_tag != "", "Name tag can't be an empty stirng"
        channels_in = in_dims[2]
        channels_out = out_dims[2]
        image_size = (in_dims[0], in_dims[1])
        intermediate_dim = (None, *image_size, filters)

        self.bn_in = BatchNormalization(axis=3, dtype=tf.float64,
                                        fused=False, name="BN_in")
        self.bn_in.build((None, *image_size, channels_in))
        self.conv_in = Conv2D(filters, kernel_size=(3, 3), padding="same",
                              data_format="channels_last", activation=None,
                              use_bias=True, dtype=tf.float64, name="Conv_in")
        self.conv_in.build((None, *image_size, channels_in * 2))
        self.conv_skip = Conv2D(filters, kernel_size=(1, 1), padding="same",
                                data_format="channels_last", activation=None,
                                use_bias=True, dtype=tf.float64,
                                name="Conv_skip_0")
        self.conv_skip.build((None, *image_size, filters))
        self.res_blocks = [ResidualConvBlock(intermediate_dim, f"i",
                                             activation, filters)
                           for i in range(nr_res_blocks)]

        self.skips = [Conv2D(filters, kernel_size=(3, 3), padding="same",
                             data_format="channels_last",
                             activation=None, use_bias=True,
                             dtype=tf.float64, name=f"Conv_skip_{i + 1}")
                      for i in range(nr_res_blocks)]
        for skip in self.skips:
            skip.build((None, *image_size, filters))
        self.bn_out = BatchNormalization(axis=3, dtype=tf.float64,
                                         fused=False, name="BN_in")
        self.bn_out.build((None, *image_size, filters))
        self.conv_out = Conv2D(channels_out, kernel_size=(1, 1),
                               padding="same", data_format="channels_last",
                               activation=None, use_bias=True,
                               dtype=tf.float64, name="Conv_out")
        self.conv_out.build((None, *image_size, filters))

    @tf.function
    def call(self, x, step, training=False):
        with tf.name_scope(f"ResNet/{self.name_tag}") as scope:
            x = self.bn_in(x)
            x = tf.concat([x, -x], 3)
            x = self.activation(x)
            x = self.conv_in(x)
            x_skip = self.conv_skip(x)

            for block, skip in zip(self.res_blocks, self.skips):
                x = block(x)
                x_skip += skip(x)

            x = self.bn_out(x_skip)
            x = self.activation(x)
            x = self.conv_out(x)

            if training and step is not None:
                self.log_weights(step)

        return x

    def log_weights(self, step):
        name = self.bn_in.name
        weights = self.bn_in.weights
        tf.summary.histogram(f"{name}/gamma", weights[0], step=step)
        tf.summary.histogram(f"{name}/beta", weights[1], step=step)

        name = self.conv_in.name
        weights = self.conv_in.weights
        tf.summary.histogram(f"{name}/kernel", weights[0], step=step)
        tf.summary.histogram(f"{name}/bias", weights[1], step=step)

        name = self.conv_skip.name
        weights = self.conv_skip.weights
        tf.summary.histogram(f"{name}/kernel", weights[0], step=step)
        tf.summary.histogram(f"{name}/bias", weights[1], step=step)

        for conv in self.skips:
            name = conv.name
            weights = conv.weights
            tf.summary.histogram(f"{name}/kernel", weights[0], step=step)
            tf.summary.histogram(f"{name}/bias", weights[1], step=step)

        name = self.bn_out.name
        weights = self.bn_out.weights
        tf.summary.histogram(f"{name}/gamma", weights[0], step=step)
        tf.summary.histogram(f"{name}/beta", weights[1], step=step)

        name = self.conv_out.name
        weights = self.conv_out.weights
        tf.summary.histogram(f"{name}/kernel", weights[0], step=step)
        tf.summary.histogram(f"{name}/bias", weights[1], step=step)


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

        BN_1 = BatchNormalization(axis=3, dtype=tf.float64, fused=False,
                                  name="BN_1")
        BN_1.build(dims)
        Conv_1 = Conv2D(filters, kernel_size=(1, 1), padding="same",
                        data_format="channels_last", activation=None,
                        use_bias=False, dtype=tf.float64, name="Conv_1")
        Conv_1.build(dims)

        BN_2 = BatchNormalization(axis=3, dtype=tf.float64, fused=False,
                                  name="BN_2")
        BN_2.build(dims)
        Conv_2 = Conv2D(filters, kernel_size=(3, 3), padding='same',
                        data_format="channels_last", activation=None,
                        use_bias=False, dtype=tf.float64, name="Conv_2")
        Conv_2.build(dims)

        BN_3 = BatchNormalization(axis=3, dtype=tf.float64, fused=False,
                                  name="BN_3")
        BN_3.build(dims)
        Conv_3 = Conv2D(filters, kernel_size=(1, 1), padding="same",
                        data_format="channels_last", activation=None,
                        use_bias=True, dtype=tf.float64, name="Conv_3")
        Conv_3.build(dims)

        self.nn_layers = [BN_1, Conv_1, BN_2, Conv_2, BN_3, Conv_3]

    @tf.function
    def call(self, x, step, training=False):
        with tf.name_scope(f"ResBlock/{self.name_tag}") as scope:
            h = x
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
    out = block(x, 0, training=True)
    tf.debugging.assert_equal(out.shape, x.shape,
                              "shape mismatch in res block")
    print(block.summary())


def test_residual_network():
    """ Unit test for the residual network."""
    in_dims = (10, 20, 8)
    out_dims = (10, 20, 16)
    name_tag = "test"
    activation = "relu"
    filters = 32
    residual_blocks = 3
    x = tf.zeros((2, 10, 20, 8), dtype=tf.float64)

    model = ResNet(in_dims, out_dims, name_tag, residual_blocks, activation,
                   filters)
    out = model(x, 0, training=True)
    tf.debugging.assert_equal(out.shape, (2, *out_dims),
                              "Shape mismatch in resnet")
    print(model.summary())

if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')
    test_residual_block()
    test_residual_network()
    print("All assertions passed, test successful")
