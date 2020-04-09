import numpy as np
import tensorflow as tf
from fc_net import FC_net
from res_net import ResNet
from tensorflow.keras import Model


class CouplingLayers(Model):
    """ Implementation of the coupling layers of the RealNVP."""

    def __init__(self, dims, name_tag, filters, scale_idx, num_scales,
                 n_blocks=3, activation='relu', architecture_type="FC_net",
                 debug=False):
        """
        Parameters
        ----------
        in_dims : (int, int, int)
            Shape of the input objects.

        n_flows : int
            Half of the number of flows in the model. This will be doubled
            with alternatingly inverted masks.
        """

        super().__init__(name=name_tag)

        self.is_last_block = scale_idx == num_scales - 1

        mask_cb = self.get_checkerboard_mask(dims)
        mask_ch = self.get_channel_mask(dims)

        self.in_couplings = [
            Coupling(dims, filters, "Coupling_layer_1", n_blocks,
                     activation, mask_cb, architecture_type, debug),
            Coupling(dims, filters, "Coupling_layer_2", n_blocks,
                     activation, 1 - mask_cb, architecture_type, debug),
            Coupling(dims, filters, "Coupling_layer_3", n_blocks,
                     activation, mask_cb, architecture_type, debug)
        ]

        if self.is_last_block:
            self.in_couplings.append(
                Coupling(dims, dims, "Coupling_layer_4", n_blocks, activation,
                         1 - mask_cb, architecture_type, debug)
            )
        else:
            filters *= 2
            dims = dims[:-1] + (4 * dims[-1], )
            self.out_couplings = [
                Coupling(dims, filters, "Coupling_layer_4", n_blocks,
                         activation, mask_ch, architecture_type, debug),
                Coupling(dims, filters, "Coupling_layer_5", n_blocks,
                         activation, 1 - mask_ch, architecture_type, debug),
                Coupling(dims, filters, "Coupling_layer_6", n_blocks,
                         activation, mask_ch, architecture_type, debug)
            ]

            name_tag = "?"  # TODO
            dims = dims[:-1] + (dims[-1] // 2, )
            self.next_couplings = CouplingLayers(dims, name_tag, filters,
                                                 scale_idx + 1, n_blocks,
                                                 activation, architecture_type,
                                                 debug)

    @tf.function
    def call(self, z, ldj, reverse=False, training=False):
        if not reverse:
            for coupling in self.in_couplings:
                z, ldj = coupling(z, ldj, training=training)

            if not self.is_last_block:
                z = self.squeeze(z)
                for coupling in self.out_couplings:
                    z, ldj = coupling(z, ldj, training=training)
                z, z_split = tf.split(z, 2, axis=-1)
                z, ldj = self.next_couplings(z, ldj, training=training)
                z = tf.concat([z, z_split], axis=-1)
                z = self.squeeze(z, reverse=True)

        else:
            if not self.is_last_block:
                z = self.squeeze(z)
                z, z_split = tf.split(z, 2, axis=-1)
                z, ldj = self.next_couplings(z, ldj, reverse=True,
                                             training=training)
                z = tf.concat([z, z_split], axis=-1)
                for coupling in self.out_couplings:
                    z, ldj = coupling(z, ldj, reverse=True, training=training)
                z = zelf.squeeze(z, reverse=True)

            for coupling in self.in_couplings:
                z, ldj = coupling(z, ldj, reverse=True, training=training)

        return z, ldj

    @staticmethod
    def get_checkerboard_mask(dims):
        """ Create checkerboard type mask."""
        size = tf.reduce_prod(dims)
        mask = np.zeros((size))
        for i in range(0, size // 2, 2):
            mask[i] = 1
        mask = np.reshape(mask, dims)
        return tf.convert_to_tensor(mask, dtype=tf.float64)

    @staticmethod
    def get_channel_mask(dims):
        """ Create channel mask."""
        dims = dims[:-1] = (dims[-1] // 2, )
        zero_half = tf.zeros(dims, dtype=tf.float64)
        one_half = tf.ones(dims, dtype=tf.float64)
        return tf.concat([zero_half, one_half], axis=-1)

    @staticmethod
    def squeeze(z, reverse=False):
        """ Squeezes the input to quadruple the number of channels.

        For each channel, it divides the input into subsquares of 2x2xc,
        then reshapes them into subsquares of 1x1x4c. The squeezing operation
        transforms a sxsxc tensor into and (s/2)x(s/2)x4c tensor.

        This squeezing is not the same as a reshape where parts of feature maps
        get stacked in the channel dimensions. See the paper for full details:
        https://arxiv.org/abs/1605.08803

        Apparently you can also do this by first reshaping to a rank-6 tensor,
        then transposing in a weird order and then reshaping to the desired
        end shape. Can be reversed in the same way

        Parameters
        ----------
        z : tensor
            The tensor that will be squeezed.

        reverse : bool
            Flag to invert the squeezing operation.

        Returns
        -------
        z : tensor
            The squeezed version of the input tensor.
        """

        if not reverse:
            z = tf.concat([z[:, 0::2, 0::2], z[:, 0::2, 1::2],
                           z[:, 1::2, 0::2], z[:, 1::2, 1::2]], axis=3)
        else:
            shape = z.shape
            h = shape[1]
            w = shape[2]
            c = shape[3]
            z = tf.concat([tf.concat([tf.reshape(z[:, i, j], (1, 2, 2, 1))
                                      for j in range(w)], axis=2)
                           for i in range(h)], axis=1)
        return z


class Coupling(Model):
    """ Single coupling layer."""

    def __init__(self, in_dims, filters, name_tag, n_blocks, activation,
                 mask, architecture_type="FC_net", debug=False):
        """
        Parameters
        ----------
        in_dims : (int, int, int)
            The input dimensions of the coupling layers

        filters : int
            The number of filters used when using ResNet as architecture type.
            If the architecture type = Fc_net, this variable is used to pass
            the number of hidden nodes in each hidden layer.

        name_tag : str
            The name of this coupling layer. Used when printing the model and
            for logging the weights in tensorboard.

        n_blocks : int
            Either the number of blocks in ResNet atchitectures or the number
            of hidden layers in FC_net architectures.

        activation : str
            The non-linear activation function used in the neural networks.

        mask : tensor
            Mask used to make the split in the input variable.
        """
        super().__init__(name=name_tag)
        self.mask = mask
        self.name_tag = name_tag

        if architecture_type == "FC_net":
            self.nn = FC_net(in_dims, 2 * in_dims, name_tag, n_blocks,
                             filters, activation, debug)
        if architecture_type == "ResNet":
            out_dims = in_dims[:-1] + (2 * in_dims[-1], )
            self.nn = ResNet(in_dims, out_dims, name_tag, n_blocks,
                             activation, filters, debug)

        weights = self.nn.layers[-1].weights
        self.nn.layers[-1].set_weights([tf.zeros_like(weights[0]),
                                        tf.zeros_like(weights[1])])

    @tf.function
    def call(self, z, ldj, reverse=False, training=False):
        """ Evaluation of a single coupling layer."""
        with tf.name_scope(f"Coupling/{self.name_tag}") as scope:
            network_forward = self.nn(z * self.mask, training=training)
            log_scale, translation = tf.split(network_forward, 2, axis=-1)

            if not reverse:
                scale = tf.math.exp(log_scale)
                z = self.mask * z + (1 - self.mask) * (z * scale + translation)
                ldj += tf.reduce_sum(log_scale * (1 - self.mask),
                                     axis=tf.range(1, tf.rank(log_scale)))
            else:
                scale = tf.math.exp(-log_scale)
                z = z * self.mask + ((z - translation) * scale)\
                    * (1 - self.mask)
                ldj = ldj  # TODO dit klopt niet

        return z, ldj


def test_coupling():
    """ Unit test for single coupling layer."""
    batch_size = 4
    name_tag = "test"
    n_blocks = 3
    activation = "relu"
    filters = 32

    dims = 100
    x = tf.ones((batch_size, dims), dtype=tf.float64)
    ldj = tf.zeros((batch_size), dtype=tf.float64)

    mask = CouplingLayers.get_mask(dims)
    coupling = Coupling(dims, dims, name_tag, n_blocks, activation,
                        mask, architecture_type="FC_net")
    z, ldj = coupling(x, ldj, training=True)
    x_recon, ldj = coupling(z, ldj, reverse=True, training=True)
    tf.debugging.assert_near(x, z, message="Coupling does not init close "
                                           "to identity.")
    tf.debugging.assert_near(x, x_recon, message="Inverse of coupling "
                                                 "incorrect")
    print(coupling.summary())

    dims = (15, 15, 3)
    x = tf.ones((batch_size, *dims), dtype=tf.float64)
    ldj = tf.zeros((batch_size), dtype=tf.float64)

    mask = CouplingLayers.get_mask(dims)
    coupling = Coupling(dims, dims, name_tag, n_blocks, activation,
                        mask, architecture_type="ResNet")
    z, ldj = coupling(x, ldj, training=True)
    x_recon, ldj = coupling(z, ldj, reverse=True, training=True)
    tf.debugging.assert_near(x, z, message="Coupling does not init close "
                                           "to identity.")
    tf.debugging.assert_near(x, x_recon, message="Inverse of coupling "
                                                 "incorrect")
    print(coupling.summary())


def test_coupling_layers():
    batch_size = 4
    name_tag = "test"
    n_blocks = 3
    activation = "relu"
    filters = 32

    dims = 100
    x = tf.ones((batch_size, dims), dtype=tf.float64)
    ldj = tf.zeros((batch_size), dtype=tf.float64)
    coupling = CouplingLayers(dims, dims, name_tag, n_blocks, activation,
                              architecture_type="FC_net", n_flows=3,
                              debug=True)
    z, ldj = coupling(x, ldj, training=True)
    x_recon, ldj = coupling(z, ldj, reverse=True, training=True)
    tf.debugging.assert_near(x, z, message="Coupling does not init close "
                                           "to identity.")
    tf.debugging.assert_near(x, x_recon, message="Inverse of coupling "
                                                 "incorrect")
    print(coupling.summary())

    dims = (15, 15, 3)
    x = tf.ones((batch_size, *dims), dtype=tf.float64)
    ldj = tf.zeros((batch_size), dtype=tf.float64)
    coupling = CouplingLayers(dims, dims, name_tag, n_blocks, activation,
                              architecture_type="ResNet", n_flows=3,
                              debug=True)
    z, ldj = coupling(x, ldj, training=True)
    x_recon, ldj = coupling(z, ldj, reverse=True, training=True)
    tf.debugging.assert_near(x, z, message="Coupling does not init close "
                                           "to identity.")
    tf.debugging.assert_near(x, x_recon, message="Inverse of coupling "
                                                 "incorrect")
    print(coupling.summary())


if __name__ == "__main__":
    tf.keras.backend.set_floatx("float64")
    test_coupling()
    test_coupling_layers()
    print("All assertions passed, test successful")
