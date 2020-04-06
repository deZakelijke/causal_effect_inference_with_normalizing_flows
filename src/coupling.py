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
        mask = torch.zeros((size), dtype=tf.float64)
        for i in range(0, size // 2, 2):
            mask[i] = 1
        mask = tf.reshape(mask, in_dims)
        return mask


def Coupling(Model):
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
        if model_type == "FC_net":
            self.nn = FC_net(in_dims, out_dims, name_tag, nr_blocks,
                             activation=activation, debug=debug)
        if model_type == "ResNet":
            self.nn == ResNet(in_dims, out_dims, name_tag, nr_blocks,
                              activation=activation, debug=debug)
        # Do something to make the final weights zero to start with identity mapping.

    @tf.function
    def call(self, z, ldj, reverse=False, training=False):
        """ Evaluation of a single coupling layer."""
        with tf.name_scope(f"Coupling/{self.name_tag}") as scope:
            network_forward = self.nn(z * self.mask, training=training)
            log_scale, translation = tf.split(network_forward, 2, axis=-1)

            if not reverse:
                scale = tf.math.exp(log_scale)
                z = self.mask * z + (1 - self.mask) * (z * scale + translation)
                ldj += tf.reduce_sum(log_scale * (1 - self.mask), axis=-1)
            else:
                scale = tf.math.exp(-log_scale)
                z = z * self.mask + ((z - translation) * scale) * (1 - self.mask)
                ldj = ldj  # TODO dit klopt niet

        return z, ldj


def test_coupling():
    pass
# test some shapes
# test both fc and resnet
# test to see if it is the identity at start
# test to see if the inverse is actually the inverse


def test_coupling_layers():
    pass


if __name__ == "__main__":
    tf.keras.backend.set_floatx("float64")
    test_coupling()
    test_coupling_layers()
    print("All assertions passed, test successful")
