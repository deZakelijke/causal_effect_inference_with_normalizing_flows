import numpy as np
import tensorflow as tf
from fc_net import FC_net
from res_net import ResNet
from tensorflow.keras import Model


class CouplingLayers(Model):
    """ Implementation of the coupling layers of the RealNVP."""

    def __init__(self, in_dims, out_dims, nr_blocks=3, activation='relu',
                 model_type="FC_net", n_flows=3, debug=False):
        """
        Parameters
        ----------
        in_dims : (int, int, int)
            Shape of the input objects.

        n_flows : int
            Half of the number of flows in the model. This will be doubled
            with alternatingly inverted masks.
        """

        super().__init()

        mask = self.get_mask(in_dims)
        self.layers = []
        for i in range(n_flows):
            self.layers.append(Coupling(in_dims, out_dims,
                                        f"Coupling_layer_{i * 2 + 1}",
                                        nr_blocs, activation, mask, model_type,
                                        debug=debug))
            self.layers.append(Coupling(in_dims, out_dims,
                                        f"Coupling_layer_{i * 2}",
                                        nr_blocks, activation, 1 - mask,
                                        model_type, debug=debug))

    @tf.function
    def call(self, z, ldj, reverse=False, training=False):
        # TODO I have to do some channel stacking here
        if not reverse:
            for layer in self.layers:
                z, ldj = layer(z, ldj, training=training)
        else:
            for layer in reversed(self.layers):
                z, ldj = layer(z, ldj, reverse=True, training=training)

        return z, ldj

    @staticmethod
    def get_mask(in_dims):
        size = tf.reduce_prod(in_dims)
        mask = np.zeros((size))
        for i in range(0, size // 2, 2):
            mask[i] = 1
        mask = np.reshape(mask, in_dims)
        return tf.convert_to_tensor(mask, dtype=tf.float64)


class Coupling(Model):
    """ Single coupling layer."""

    def __init__(self, in_dims, out_dims, name_tag, nr_blocks, activation,
                 mask, model_type="FC_net", debug=False):
        """
        Parameters
        ----------
        mask : tensor
        """
        super().__init__()
        self.mask = mask
        self.name_tag = name_tag

        if model_type == "FC_net":
            self.nn = FC_net(in_dims, 2 * out_dims, name_tag, nr_blocks,
                             activation=activation, debug=debug)
        if model_type == "ResNet":
            out_dims = out_dims[:-1] + (2 * out_dims[-1], )
            self.nn = ResNet(in_dims, out_dims, name_tag, nr_blocks,
                             activation=activation, debug=debug)

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
    nr_blocks = 3
    activation = "relu"
    filters = 32

    dims = 100
    x = tf.ones((batch_size, dims), dtype=tf.float64)
    ldj = tf.zeros((batch_size), dtype=tf.float64)
    model_type = "FC_net"

    mask = CouplingLayers.get_mask(dims)
    coupling = Coupling(dims, dims, name_tag, nr_blocks, activation,
                        mask, model_type=model_type)
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
    model_type = "ResNet"

    mask = CouplingLayers.get_mask(dims)
    coupling = Coupling(dims, dims, name_tag, nr_blocks, activation,
                        mask, model_type=model_type)
    z, ldj = coupling(x, ldj, training=True)
    x_recon, ldj = coupling(z, ldj, reverse=True, training=True)
    tf.debugging.assert_near(x, z, message="Coupling does not init close "
                                           "to identity.")
    tf.debugging.assert_near(x, x_recon, message="Inverse of coupling "
                                                 "incorrect")
    print(coupling.summary())


def test_coupling_layers():
    pass


if __name__ == "__main__":
    tf.keras.backend.set_floatx("float64")
    test_coupling()
    test_coupling_layers()
    print("All assertions passed, test successful")
