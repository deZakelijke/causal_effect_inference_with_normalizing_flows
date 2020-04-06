import tensorflow as tf
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

    def __init__(self):
        """
        Parameters
        ----------

        """
        pass

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
