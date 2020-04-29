import numpy as np
import tensorflow as tf
from fc_net import FC_net
from res_net import ResNet
from tensorflow.keras import Model


class CouplingLayers(Model):
    """ Implementation of the coupling layers of the RealNVP."""

    def __init__(self, dims, name_tag, filters, scale_idx, n_scales,
                 n_blocks=3, activation='relu', architecture_type="FC_net",
                 context=False, context_dims=0, debug=False):
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
        self.debug = debug
        self.name_tag = name_tag
        self.context = context
        self.is_last_block = scale_idx == n_scales - 1
        self.no_squeeze = architecture_type == "FC_net"

        if context and context_dims == 0:
            raise ValueError("Conxtex dims can't be zero if context is True.")

        mask = self.get_checkerboard_mask(dims)
        coupling_index = scale_idx * 6

        self.in_couplings = [
            Coupling(dims, filters, f"Coupling_layer_{coupling_index + 1}",
                     n_blocks, activation, mask, architecture_type, context,
                     context_dims, debug),
            Coupling(dims, filters, f"Coupling_layer_{coupling_index + 2}",
                     n_blocks, activation, 1 - mask, architecture_type,
                     context, context_dims, debug),
            Coupling(dims, filters, f"Coupling_layer_{coupling_index + 3}",
                     n_blocks, activation, mask, architecture_type, context,
                     context_dims, debug)
        ]

        if self.is_last_block:
            self.in_couplings.append(
                Coupling(dims, filters, f"Coupling_layer_{coupling_index + 4}",
                         n_blocks, activation, 1 - mask, architecture_type,
                         context, context_dims, debug)
            )
        else:
            if architecture_type == "ResNet":
                filters *= 2
                dims = (dims[0] // 2, dims[1] // 2, 4 * dims[-1])
                mask = self.get_channel_mask(dims)

            self.out_couplings = [
                Coupling(dims, filters, f"Coupling_layer_{coupling_index + 4}",
                         n_blocks, activation, 1 - mask, architecture_type,
                         context, context_dims, debug),
                Coupling(dims, filters, f"Coupling_layer_{coupling_index + 5}",
                         n_blocks, activation, mask, architecture_type,
                         context, context_dims, debug),
                Coupling(dims, filters, f"Coupling_layer_{coupling_index + 6}",
                         n_blocks, activation, 1 - mask, architecture_type,
                         context, context_dims, debug)
            ]

            name_tag = "TODO_make_better_name"  # TODO
            if architecture_type == "ResNet":
                dims = dims[:-1] + (dims[-1] // 2, )
            else:
                dims = dims // 2
            self.next_couplings = CouplingLayers(dims, name_tag, filters,
                                                 scale_idx + 1, n_scales,
                                                 n_blocks, activation,
                                                 architecture_type, context,
                                                 context_dims, debug)

    @tf.function
    def call(self, z, ldj, step, reverse=False, training=False, t=None):
        if self.debug:
            print(f"CouplingLayer: {self.name_tag}")
        if not reverse:
            for coupling in self.in_couplings:
                z, ldj = coupling(z, ldj, step, training=training, t=t)

            if not self.is_last_block:
                z = self.squeeze(z)
                for coupling in self.out_couplings:
                    z, ldj = coupling(z, ldj, step, training=training, t=t)
                z, z_split = tf.split(z, 2, axis=-1)
                z, ldj = self.next_couplings(z, ldj, step, training=training,
                                             t=t)
                z = tf.concat([z, z_split], axis=-1)
                z = self.squeeze(z, reverse=True)

        else:
            if not self.is_last_block:
                z = self.squeeze(z)
                z, z_split = tf.split(z, 2, axis=-1)
                z, ldj = self.next_couplings(z, ldj, step, reverse=True,
                                             training=training, t=t)
                z = tf.concat([z, z_split], axis=-1)
                for coupling in reversed(self.out_couplings):
                    z, ldj = coupling(z, ldj, step, reverse=True,
                                      training=training, t=t)
                z = self.squeeze(z, reverse=True)

            for coupling in reversed(self.in_couplings):
                z, ldj = coupling(z, ldj, step, reverse=True,
                                  training=training, t=t)

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
        dims = dims[:-1] + (dims[-1] // 2, )
        zero_half = tf.zeros(dims, dtype=tf.float64)
        one_half = tf.ones(dims, dtype=tf.float64)
        return tf.concat([zero_half, one_half], axis=-1)

    def squeeze(self, z, reverse=False):
        """ Squeezes the input to quadruple the number of channels.

        For each channel, it divides the input into subsquares of 2x2xc,
        then reshapes them into subsquares of 1x1x4c. The squeezing operation
        transforms a sxsxc tensor into and (s/2)x(s/2)x4c tensor.

        This squeezing is not the same as a reshape where parts of feature maps
        get stacked in the channel dimensions. See the paper for full details:
        https://arxiv.org/abs/1605.08803

        Apparently you can also do this by first reshaping to a rank-6 tensor,
        then transposing in a weird order and then reshaping to the desired
        end shape. Can be reversed in the same way.
        That's what I did. Kudos go to:
        https://github.com/chrischute/real-nvp/blob/master/util/array_util.py

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

        if self.no_squeeze:
            return z

        shape = z.shape
        b = shape[0]
        h = shape[1]
        w = shape[2]
        c = shape[3]

        if not reverse:
            z = tf.reshape(z, (b, h // 2, 2, w // 2, 2, c))
            z = tf.transpose(z, (0, 1, 3, 5, 2, 4))
            z = tf.reshape(z, (b, h // 2, w // 2, c * 4))

        else:
            z = tf.reshape(z, (b, h, w, c // 4, 2, 2))
            z = tf.transpose(z, (0, 1, 4, 2, 5, 3))
            z = tf.reshape(z, (b, h * 2, w * 2, c // 4))

        return z


class Coupling(Model):
    """ Single coupling layer."""

    def __init__(self, in_dims, filters, name_tag, n_blocks, activation,
                 mask, architecture_type="FC_net", context=False,
                 context_dims=0, debug=False):
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
        self.debug = debug
        self.mask = mask
        self.name_tag = name_tag
        self.context = context

        if context and context_dims == 0:
            raise ValueError("Conxtex dims can't be zero if context is True.")

        if architecture_type == "FC_net":
            out_dims = in_dims * 2
            in_dims += context_dims
            self.nn = FC_net(in_dims, out_dims, name_tag, n_blocks,
                             filters, activation, debug)
        if architecture_type == "ResNet":
            out_dims = in_dims[:-1] + (2 * in_dims[-1], )
            in_dims = in_dims[:-1] + (in_dims[-1] + context_dims, )
            self.nn = ResNet(in_dims, out_dims, name_tag, n_blocks,
                             filters, activation, debug)

        weights = self.nn.layers[-1].weights
        self.nn.layers[-1].set_weights([tf.zeros_like(weights[0]),
                                        tf.zeros_like(weights[1])])

    @tf.function
    def call(self, z, ldj, step, reverse=False, training=False, t=None):
        """ Evaluation of a single coupling layer."""

        if self.debug:
            print(f"Coupling: {self.name_tag}")
        with tf.name_scope(f"Coupling/{self.name_tag}") as scope:
            if self.context:
                network_in = tf.concat([z * self.mask, t], axis=-1)
            else:
                network_in = z * self.mask
            network_forward = self.nn(network_in, step, training=training)
            log_scale, translation = tf.split(network_forward, 2, axis=-1)
            log_scale = tf.math.tanh(log_scale)

            if not reverse:
                scale = tf.math.exp(log_scale)
                z = self.mask * z + (1 - self.mask) * (z * scale + translation)
                ldj += tf.reduce_sum(log_scale * (1 - self.mask),
                                     axis=tf.range(1, tf.rank(log_scale)))
            else:
                scale = tf.math.exp(-log_scale)
                z = z * self.mask + ((z - translation) * scale)\
                    * (1 - self.mask)
                ldj -= tf.reduce_sum(log_scale * (1 - self.mask),
                                     axis=tf.range(1, tf.rank(log_scale)))

        return z, ldj


def test_coupling():
    """ Unit test for single coupling layer."""
    batch_size = 8
    name_tag = "test"
    n_blocks = 4
    activation = "relu"
    filters = 128
    dims = 100

    x = tf.ones((batch_size, dims), dtype=tf.float64)
    ldj = tf.zeros((batch_size), dtype=tf.float64)

    mask = CouplingLayers.get_checkerboard_mask(dims)
    coupling = Coupling(dims, filters, name_tag, n_blocks, activation,
                        mask, architecture_type="FC_net", debug=True)
    z, ldj_out = coupling(x, ldj, training=True)
    x_recon, ldj_recon = coupling(z, ldj_out, reverse=True, training=True)
    tf.debugging.assert_near(x, z, message="Coupling does not init close "
                                           "to identity.")
    tf.debugging.assert_near(x, x_recon, message="Inverse of coupling "
                                                 "incorrect")
    tf.debugging.assert_near(ldj_recon, ldj, message="Inverse of ldj is "
                                                     "incorrect")
    print(coupling.summary())

    filters = 32
    dims = (15, 15, 4)

    x = tf.ones((batch_size, *dims), dtype=tf.float64)
    ldj = tf.zeros((batch_size), dtype=tf.float64)

    mask = CouplingLayers.get_channel_mask(dims)
    coupling = Coupling(dims, filters, name_tag, n_blocks, activation,
                        mask, architecture_type="ResNet", debug=True)
    z, ldj_out = coupling(x, ldj, 0, training=True)
    x_recon, ldj_recon = coupling(z, ldj_out, 0, reverse=True, training=True)
    tf.debugging.assert_near(x, z, message="Coupling does not init close "
                                           "to identity.")
    tf.debugging.assert_near(x, x_recon, message="Inverse of coupling "
                                                 "incorrect")
    tf.debugging.assert_near(ldj_recon, ldj, message="Inverse of ldj is "
                                                     "incorrect")
    print(coupling.summary())


def test_coupling_layers():
    batch_size = 4
    name_tag = "test_coupling_layers_fc_net"
    n_blocks = 3
    activation = "relu"
    n_scales = 2
    filters = 128
    dims = 100
    context_dims = 20

    x = tf.ones((batch_size, dims), dtype=tf.float64)
    t = tf.ones((batch_size, context_dims), dtype=tf.float64)
    ldj = tf.zeros((batch_size), dtype=tf.float64)
    coupling = CouplingLayers(dims, name_tag, filters, 0, n_scales, n_blocks,
                              activation, "FC_net", True, context_dims,
                              debug=True)
    z, ldj_out = coupling(x, ldj, 0, training=True, t=t)
    x_recon, ldj_recon = coupling(z, ldj_out, 0, reverse=True, training=True,
                                  t=t)
    tf.debugging.assert_near(x, z, message="Coupling does not init close "
                                           "to identity.")
    tf.debugging.assert_near(x, x_recon, message="Inverse of coupling "
                                                 "incorrect")
    tf.debugging.assert_near(ldj_recon, ldj, message="Inverse of ldj is "
                                                     "incorrect")
    print(coupling.summary())

    name_tag = "test_coupling_layers_resnet"
    filters = 32
    dims = (32, 32, 4)

    x = tf.ones((batch_size, *dims), dtype=tf.float64)
    ldj = tf.zeros((batch_size), dtype=tf.float64)
    coupling = CouplingLayers(dims, name_tag, filters, 0, n_scales, n_blocks,
                              activation, architecture_type="ResNet",
                              debug=True)
    z, ldj_out = coupling(x, ldj, 0, training=True)
    x_recon, ldj_recon = coupling(z, ldj_out, 0, reverse=True, training=True)
    tf.debugging.assert_near(x, z, message="Coupling does not init close "
                                           "to identity.")
    tf.debugging.assert_near(x, x_recon, message="Inverse of coupling "
                                                 "incorrect")
    tf.debugging.assert_near(ldj_recon, ldj, message="Inverse of ldj is "
                                                     "incorrect")
    print(coupling.summary())


if __name__ == "__main__":
    tf.keras.backend.set_floatx("float64")
    test_coupling()
    test_coupling_layers()
    print("All assertions passed, test successful")
