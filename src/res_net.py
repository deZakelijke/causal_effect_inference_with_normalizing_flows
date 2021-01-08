import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, activations, Sequential
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, Lambda
from tensorflow.keras.layers import Conv2DTranspose


class ResNet(Model):
    """Residual Convolutional Neural Network."""

    def __init__(self, in_dims=(16, 16, 3), out_dims=(16, 16, 3),
                 name_tag="res_net", n_layers=3, feature_maps=32,
                 activation='elu', squeeze=False, squeeze_dims=None,
                 unsqueeze=False, coord_conv=True, debug=False):
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

        n_layers : int
            The number of residual blocks in the middle of the networks.
            Each residual block has filter number of feature_maps.

        activation : str
            Activation function after every layer.

        feature_maps : int
            Number of feature_maps to use in the residual blocks.

        squeeze : bool
            Flag to add a final fully connected layer to the network that
            changes the output shape.

        squeeze_dims : int
            Size of the output after the squeeze layer.

        debug : bool
            Flag to enable debug mode. Currently unused.
        """

        super().__init__()

        self.debug = debug
        self.name_tag = name_tag
        self.activation = Activation(activation)
        self.squeeze = squeeze
        self.coord_conv = coord_conv

        assert name_tag != "", "Name tag can't be an empty stirng"
        try:
            channels_in = in_dims[2]
        except TypeError:
            channels_in = 1

        channels_out = out_dims[2]
        try:
            image_size = (in_dims[0], in_dims[1])
        except TypeError:
            image_size = (out_dims[0], out_dims[1])
        self.image_size = image_size
        intermediate_dim = (None, *image_size, feature_maps)

        self.in_layers = Sequential()
        if unsqueeze:
            assert np.sqrt(in_dims) ** 2 == in_dims
            im_height = int(np.sqrt(in_dims))
            conv_trans1 = Conv2DTranspose(channels_in,
                                          kernel_size=(3, 3),
                                          data_format="channels_last",
                                          activation=activation,
                                          use_bias=True,
                                          dtype=tf.float64,
                                          dilation_rate=8)
            conv_trans2 = Conv2DTranspose(channels_in,
                                          kernel_size=(3, 3),
                                          data_format="channels_last",
                                          activation=activation,
                                          use_bias=True,
                                          dtype=tf.float64,
                                          dilation_rate=6)
            conv_trans3 = Conv2DTranspose(channels_in,
                                          kernel_size=(3, 3),
                                          data_format="channels_last",
                                          activation=activation,
                                          use_bias=True,
                                          dtype=tf.float64,
                                          dilation_rate=8)
            conv_trans1.build((None, im_height, im_height, channels_in))
            conv_trans2.build((None, 24, 24, channels_in))
            conv_trans3.build((None, 40, 40, channels_in))
            self.in_layers.add(Reshape((im_height, im_height, channels_in)))
            self.in_layers.add(conv_trans1)
            self.in_layers.add(conv_trans2)
            self.in_layers.add(conv_trans3)

        else:
            self.in_layers.add(Lambda(tf.identity))

        if coord_conv:
            channels_in += 2

        self.bn_in = BatchNormalization(axis=3, dtype=tf.float64,
                                        fused=False, name="BN_in")
        self.bn_in.build((None, *image_size, channels_in))

        self.conv_in = Conv2D(feature_maps, kernel_size=(3, 3), padding="same",
                              data_format="channels_last", activation=None,
                              use_bias=True, dtype=tf.float64, name="Conv_in")
        self.conv_in.build((None, *image_size, channels_in * 2))

        self.conv_skip = Conv2D(feature_maps, kernel_size=(1, 1),
                                padding="same",
                                data_format="channels_last", activation=None,
                                use_bias=True, dtype=tf.float64,
                                name="Conv_skip_0")
        self.conv_skip.build((None, *image_size, feature_maps))

        self.res_blocks = [ResidualConvBlock(intermediate_dim, f"i",
                                             activation, feature_maps)
                           for i in range(n_layers)]

        self.skips = [Conv2D(feature_maps, kernel_size=(3, 3), padding="same",
                             data_format="channels_last",
                             activation=None, use_bias=True,
                             dtype=tf.float64, name=f"Conv_skip_{i + 1}")
                      for i in range(n_layers)]
        for skip in self.skips:
            skip.build((None, *image_size, feature_maps))

        self.bn_out = BatchNormalization(axis=3, dtype=tf.float64,
                                         fused=False, name="BN_in")
        self.bn_out.build((None, *image_size, feature_maps))

        conv_out = Conv2D(channels_out, kernel_size=(1, 1), padding="same",
                          data_format="channels_last", activation=None,
                          use_bias=True, dtype=tf.float64, name="Conv_out")
        conv_out.build((None, *image_size, feature_maps))
        self.out_layers = Sequential()
        self.out_layers.add(conv_out)

        if squeeze:
            fc_out = Dense(squeeze_dims, activation=None, dtype=tf.float64,
                           name="dense_out")
            fc_out.build((None, tf.reduce_prod(image_size) * channels_out))
            self.out_layers.add(self.activation)
            self.out_layers.add(Flatten())
            self.out_layers.add(fc_out)
        self.out_layers.build((None, *image_size, feature_maps))

    @tf.function
    def call(self, x, step, training=False):
        with tf.name_scope(f"ResNet/{self.name_tag}") as scope:
            x = self.in_layers(x)
            if self.coord_conv:
                x = self.add_coordinate_channels(x)
            x = self.bn_in(x)
            x = tf.concat([x, -x], 3)
            x = self.activation(x)
            x = self.conv_in(x)
            x_skip = self.conv_skip(x)

            for block, skip in zip(self.res_blocks, self.skips):
                x = block(x, step, training=training)
                x_skip += skip(x)

            x = self.bn_out(x_skip)
            x = self.activation(x)
            x = self.out_layers(x)

            # if training and step is not None and step % 50 == 0:
            #     self.log_weights(step)

        return x

    def add_coordinate_channels(self, x):
        """ Adding coord conv channels to the images.

        https://github.com/uber-research/CoordConv/blob/master/CoordConv.py
        """
        batch_size = x.shape[0]

        xx_ones = tf.ones((batch_size, self.image_size[0], 1),
                          dtype=tf.float64)
        xx_range = tf.expand_dims(tf.range(self.image_size[1],
                                  dtype=tf.float64), 0)
        xx_range = tf.expand_dims(tf.tile(xx_range, [batch_size, 1]), 1)
        xx_channel = xx_ones * xx_range
        xx_channel = tf.expand_dims(xx_channel, -1)

        yy_ones = tf.ones((batch_size, 1, self.image_size[1]),
                          dtype=tf.float64)
        yy_range = tf.expand_dims(tf.range(self.image_size[0],
                                  dtype=tf.float64), 0)
        yy_range = tf.expand_dims(tf.tile(yy_range, [batch_size, 1]), -1)
        yy_channel = tf.expand_dims(yy_ones * yy_range, -1)

        xx_channel = xx_channel / (self.image_size[0] - 1) * 2 - 1
        yy_channel = yy_channel / (self.image_size[1] - 1) * 2 - 1
        return tf.concat([x, xx_channel, yy_channel], -1)

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

        name = self.out_layers.layers[0].name
        weights = self.out_layers.layers[0].weights
        tf.summary.histogram(f"{name}/kernel", weights[0], step=step)
        tf.summary.histogram(f"{name}/bias", weights[1], step=step)


class ResidualConvBlock(Model):
    """ Single block of a convolutional neural network with a skip connection.

    Contains a convolution with nonlinearity, maxpooling, batch norm and
    a skip connection.
    """

    def __init__(self, dims, name_tag, activation, feature_maps):
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

        feature_maps : int
            Number of feature_maps in the convolutional block.
        """

        super().__init__(dtype=tf.float64)
        self.name_tag = name_tag
        self.activation = eval(f"activations.{activation}")

        BN_1 = BatchNormalization(axis=3, dtype=tf.float64, fused=False,
                                  name="BN_1")
        BN_1.build(dims)
        Conv_1 = Conv2D(feature_maps, kernel_size=(1, 1), padding="same",
                        data_format="channels_last", activation=None,
                        use_bias=False, dtype=tf.float64, name="Conv_1")
        Conv_1.build(dims)

        BN_2 = BatchNormalization(axis=3, dtype=tf.float64, fused=False,
                                  name="BN_2")
        BN_2.build(dims)
        Conv_2 = Conv2D(feature_maps, kernel_size=(3, 3), padding='same',
                        data_format="channels_last", activation=None,
                        use_bias=False, dtype=tf.float64, name="Conv_2")
        Conv_2.build(dims)

        BN_3 = BatchNormalization(axis=3, dtype=tf.float64, fused=False,
                                  name="BN_3")
        BN_3.build(dims)
        Conv_3 = Conv2D(feature_maps, kernel_size=(1, 1), padding="same",
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
            # if training and step is not None and step % 50 == 0:
            #     self.log_weights(step)
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
    feature_maps = 32
    x = tf.zeros((2, 10, 20, 32), dtype=tf.float64)

    block = ResidualConvBlock(dims, name_tag, activation, feature_maps)
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
    feature_maps = 32
    residual_blocks = 3
    x = tf.zeros((2, 10, 20, 8), dtype=tf.float64)

    model = ResNet(in_dims, out_dims, name_tag, residual_blocks, feature_maps,
                   activation, squeeze=True, squeeze_dims=1, coord_conv=True,
                   debug=True)
    out = model(x, 0, training=True)
    tf.debugging.assert_equal(out.shape, (2, 1),
                              "Shape mismatch in resnet")
    print(model.summary())


if __name__ == "__main__":
    set_vdc = tf.config.experimental.set_virtual_device_configuration
    vdc = tf.config.experimental.VirtualDeviceConfiguration
    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            # set_vdc(gpu, [vdc(memory_limit=4096)])
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs, ", len(logical_gpus),
                  "Logical GPUs")

    tf.keras.backend.set_floatx('float64')
    test_residual_block()
    test_residual_network()
    print("All assertions passed, test successful")
