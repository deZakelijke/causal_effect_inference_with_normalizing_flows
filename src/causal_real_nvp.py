import math
import tensorflow as tf
import time

from coupling import CouplingLayers
from tensorflow.keras import Model


class CausalRealNVP(Model):
    """ Adaptation of the RealNVP for causal inference.

    This class is an adaption of the RealNVP model for causal inference.
    It has two observed states (images for example) and an action/intervention
    that is one of the causes for the second state. There also can/should be
    latent confounding between the second state and the action. The first state
    serves as a noisy and incomplete proxy for this latent confounder.
    The model is capable of inferring the latent state that is a cause for both
    the first and the second state, given the two states and the action that
    connects them.
    """

    def __init__(self, dims, intervention_dims, name_tag, filters, n_scales,
                 n_blocks=3, activation="relu", architecture_type="FC_net",
                 debug=False):
        """
        Parameters
        ----------

        """
        super().__init__(name=name_tag)
        self.dims = dims
        self.debug = debug

        with tf.name_scope(f"RealNVP/{name_tag}") as scope:
            self.flow_x = CouplingLayers(dims, "flow_x", filters, 0, n_scales,
                                         n_blocks, activation,
                                         architecture_type, debug=debug)

            self.flow_y = CouplingLayers(dims, "flow_y", filters, 0, n_scales,
                                         n_blocks, activation,
                                         architecture_type, context=True,
                                         context_dims=intervention_dims,
                                         debug=debug)
            # TODO I have to adjust flow_y in such a way that it also uses the
            # context variable
            if architecture_type == "FC_net":
                dims_z = dims * 2
            else: 
                dims_z = dims[:-1] + (2 * dims[-1], )
            self.flow_z = CouplingLayers(dims_z, "flow_z", filters, 0, n_scales,
                                         n_blocks, activation,
                                         architecture_type, debug=debug)

    def dequantize(self, z):
        """ Dequantisation of the input data.

        Causes the data to not lie on exact integer values. Only use this
        when the data has a finite discrete set of possible values.
        For instance images where each pixel is in [0, 255]. The dequantisation
        operation makes the data lie on the continuous interval [0, 256) which
        prevents the model from collapsing its probability density to a set of
        points instead of being a smooth function.

        Parameters
        ----------
        z : tensor
            A tensor of any shape that has discrete values. The data type can
            be any type

        Returns
        -------
        z : tensor
            The new tensor of the same shape as the input.
        """
        z = tf.cast(z, dtype=tf.float64)
        return z + tf.random.uniform(z.shape, dtype=tf.float64)

    def logit_normalize(self, z, ldj, reverse=False, alpha=1e-6):
        """ Inverse sigmoid normalisation.

        Parameters
        ----------
        z : tensor
            The data that will be normalised. Alle values should lie between
            zero and one.

        ldj : tensor
            The log determinant of the Jacobian needed to define the
            probability of z after the normalisation.

        reverse : bool
            Flag to use the inverse of the normalisation, which is a regular
            sigmoid transform.

        alpha : float
            Small value to prevent the calculation from yielding NaNs when z
            is very close to zero or one.
        """

        maximal = tf.constant(256, dtype=tf.float64)
        if not reverse:
            z /= maximal
            ldj -= tf.math.log(maximal) * tf.cast(tf.reduce_prod(z.shape[1:]),
                                                  tf.float64)
            z = z * (1 - alpha) + alpha * 0.5
            ldj += tf.reduce_sum(-tf.math.log(z) - tf.math.log(1 - z),
                                 axis=tf.range(1, tf.rank(z)))
            z = tf.math.log(z) - tf.math.log(1 - z)

        else:
            # TODO is this order really the inverse?
            ldj -= tf.reduce_sum(-tf.math.log(z) - tf.math.log(1 - z),
                                 axis=tf.range(1, tf.rank(z)))
            z = tf.math.sigmoid(z)
            z *= maximal
            ldj += tf.math.log(maximal) * tf.cast(tf.reduce_prod(z.shape[1:]),
                                                  tf.float64)

        return z, ldj

    @staticmethod
    def log_prior(z):
        """ The prior log probability of a standard Gaussian: N(z|mu=0, sig=1)

        Parameters
        ----------
        z : tensor
            The tensor of which the prior probability should be calculated.
            Can be any shape
        """

        pi = tf.constant(math.pi, dtype=tf.float64)
        norm_term = tf.math.log(1 / tf.math.sqrt(2 * pi))
        logp = norm_term - 0.5 * tf.math.square(z)
        return tf.reduce_sum(logp)

    @tf.function
    def call(self, x, t, y, step, training=False):
        """ Infers the latent confounder and the log-likelihood of the input.

        Calculates the latent confouder z by mapping the input through several
        coupling layers. It also calculates the log-likelihood of the input
        triplet by adding the log-likelihood of z to the log determinant
        Jacobian of the mapping.

        Parameters
        ----------
        x : tensor
            The first state of the triplet.

        t : tensor
            The action that causes the second state.

        y : tensor
            The second state, that is caused by both t and z.

        step : int
            Training step index used to index the logging of weights and
            other statistics in Tensorboard.

        training : bool
            Flag to set the model to training mode.

        Returns
        -------
        log_pxy : float
            The log likelihood of both x and y.
        z : tensor
            The latent state corresponding to the input triplet.
        """

        # TODO name scope
        ldj = tf.zeros(x.shape[0], dtype=tf.float64)

        x = self.dequantize(x)
        y = self.dequantize(y)
        x, ldj = self.logit_normalize(x, ldj)
        y, ldj = self.logit_normalize(y, ldj)

        x_intermediate, ldj = self.flow_x(x, ldj, step, training=training)
        y_intermediate, ldj = self.flow_y(y, ldj, step, training=training, t=t)
        # TODO Flow y with context t

        z = tf.concat([x_intermediate, y_intermediate], axis=-1)
        z, ldj = self.flow_z(z, ldj, step, training=training)

        log_pz = self.log_prior(z)
        log_pxy = log_pz + ldj

        return log_pxy, z

    def _reverse_flow(self, z, t, ldj):
        """ Reverse of flow"""

        z, ldj = self.flow_z(z, ldj, None, reverse=True)
        x, y = tf.split(z, 2, axis=-1)
        y, ldj = self.flow_y(y, ldj, None, reverse=True, t=t)
        x, ldj = self.flow_x(x, ldj, None, reverse=True)

        y, ldj = self.logit_normalize(y, ldj, reverse=True)
        x, ldj = self.logit_normalize(x, ldj, reverse=True)

        return x, y, ldj

    def sample(self, n_samples):
        shape = (n_samples, self.dims[0], self.dims[1], self.dims[2] * 2)
        z = tf.random.normal(shape)
        ldj = tf.zeros(n_samples, dtype=tf.float64)
        x, y, ldj = self._reverse_flow(z, ldj, 0)
        return x, y


def test_model():
    ds = tfds.load('cifar10')

    # dims = (32, 32, 3)
    dims = 100
    intervention_dims = 20
    name_tag = "test"
    filters = 32
    n_scales = 3
    n_blocks = 4
    activation = "relu"
    # architecture_type = "ResNet"
    architecture_type = "FC_net"
    batch_size = 8

    # for data in ds['train'].batch(2 * batch_size).take(1):
    #     images = data['image']
    #     x = images[:batch_size]
    #     y = images[batch_size:]

    x = tf.ones((batch_size, dims), dtype=tf.float64)
    y = tf.ones((batch_size, dims), dtype=tf.float64)
    t = tf.ones((batch_size, intervention_dims), dtype=tf.float64)

    # TODO writ test for logit normalise
    start_time = time.time()
    model = CausalRealNVP(dims, intervention_dims, name_tag, filters, n_scales,
                          n_blocks,
                          activation, architecture_type, debug=True)
    middle_time = time.time()
    log_pxy, z = model(x, t, y, 0, training=True)
    end_time = time.time()

    x_recon, y_recon, _ = model._reverse_flow(z, t, log_pxy)
    tf.debugging.assert_near(tf.cast(x, tf.float64), x_recon,
                             atol=1.0 + 1e-5, rtol=1e-5,
                             message="Reconstructing of x incorrect")
    tf.debugging.assert_near(tf.cast(y, tf.float64), y_recon,
                             atol=1.0 + 1e-5, rtol=1e-5,
                             message="Reconstruction of y incorrect")

    print(model.summary())
    print(f"Time to init model: {middle_time - start_time}")
    print(f"Time to do forward pass: {end_time - middle_time}")

if __name__ == "__main__":
    import tensorflow_datasets as tfds
    tf.keras.backend.set_floatx("float64")
    test_model()
    print("All assertions passed, test successful")
