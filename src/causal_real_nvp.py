import tensorflow as tf

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

    def __init__(self, dims, name_tag, n_flows, activation="relu",
                 n_blocks=3, architecture_type="FC_net", debug=False):
        """
        Parameters
        ----------

        """
        super().__init__(name=name_tag)

        if len(dims) > 1:
            # We need to do layer stacking
            # This is actually also a flag of the CouplingLayers class

        with tf.name_scope(f"RealNVP/{name_tag}") as scope:
            self.flow_x = CouplingLayers(dims, "flow_x", n_blocks, activation,
                                         architecture_type, n_flows, debug)
            self.flow_y = CouplingLayers(dims, "flow_y", n_blocks, activation,
                                         architecture_type, n_flows, debug)
            # TODO I have to adjust flow_y in such a way that it also uses the
            # context variable
            dims = dims[:-1] + (2 * dims[-1], )
            self.flow_z = CouplingLayers(dims, "flow_z", n_blocks, activation,
                                         architecture_type, n_flows, debug)

    def dequantize(self z):
        return z + tf.random.uniform(z.shape, dtype=tf.float64)

    def logit_normalize(self, z, ldj, reverse=False, alpha=1e-5):
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

        if not reverse:
            z /= 256
            ldj -= tf.math.log(256) * tf.reduce_prod(z.shape[1:])
            z = z * (1 - alpha) + alpha * 0.5
            ldj += tf.reduce_sum(-tf.math.log(z) - tf.math.log(1 - z), axis=1)
            # TODO shouldn't the axis of the sum here be all axes except the first?
            z = tf.math.log(z) - tf.math.log(1 - z)

        else:
            # TODO is this order really the inverse?
            ldj -= tf.reduce_sum(-tf.math.log(z) - tf.math.log(1 - z), axis=1)
            z = tf.math.sigmoid(z)
            z *= 256
            ldj += tf.math.log(256) * tf.reduce_prod(z.shape[1:])

        return z, logdet

    @staticmethod
    def log_prior(z):
        """ The prior log probability of a standard Gaussian: N(z|mu=0, sig=1)

        Parameters
        ----------
        z : tensor
            The tensor of which the prior probability should be calculated.
            Can be any shape
        """

        norm_term = tf.math.log(1 / tf.math.sqrt(2 * tf.math.pi))
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
        z : tensor
            The latent state corresponding to the input triplet.
        log_pxy : float
            The log likelihood of both x and y.
        """

        ldj = tf.zeros(x.shape[0], dtype=tf.float64)

        x_input = self.dequantize(x)
        y_input = self.dequantize(y)

        # Logit normalize
        # Flow x
        # Flow y with context t
        # Concat
        # Flow z

        # Log prior z
        # Log pxy = log prior + ldj
