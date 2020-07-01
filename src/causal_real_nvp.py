import math
import tensorflow as tf
import time

from coupling import CouplingLayers
from fc_net import FC_net
from res_net import ResNet
from tensorflow.keras import Model
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.layers import Dense


class CRNVP(Model):
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

    def __init__(
        self,
        x_dims=(50, 50, 3),
        x_cat_dims=(50, 50, 0),
        x_cont_dims=(50, 50, 3),
        t_dims=16,
        y_dims=1,
        z_dims=(50, 50, 6),
        category_sizes=0,
        t_loss=CategoricalCrossentropy(),
        y_loss=MeanSquaredError(),
        name_tag="no_name",
        feature_maps=32,
        n_scales=2,
        n_layers=3,
        activation="relu",
        architecture_type="FC_net",
        log_steps=10,
        debug=False,
        **_
    ):
        """
        Parameters
        ----------

        """

        super().__init__(name=name_tag)
        self.debug = debug
        self.log_steps = log_steps
        self.t_dims = t_dims
        self.t_loss = t_loss
        self.y_dims = y_dims
        self.y_loss = y_loss
        network = eval(architecture_type)

        self.annealing_factor = 1e-8

        if architecture_type == "FC_net":
            self.x_dims = (x_dims, )
            # y_dims *= 2
            # self.y_dims = y_dims
            z_dims = x_dims + y_dims * 2
            self.dims_split = [x_dims, y_dims * 2]
        else:
            self.x_dims = x_dims
            # z_dims = x_dims[:-1] + (x_dims[-1] + y_dims[-1], )
            z_dims = tf.reduce_prod(x_dims) + y_dims * 2
            self.dims_split = [tf.reduce_prod(x_dims), y_dims * 2]

        with tf.name_scope(f"RealNVP/{name_tag}") as scope:
            self.flow_x = CouplingLayers(x_dims, "flow_x", feature_maps, 0,
                                         n_scales, n_layers, activation,
                                         architecture_type, debug=debug)
            # TODO make y have its own shape. Check shape of t
            # Also adapt the flow length to the dimensionality of y
            # But what if y has an uneven number of dimensions or if it
            # has just one dimension?
            n_scales = 1
            self.flow_y = CouplingLayers(y_dims, "flow_y", feature_maps, 0,
                                         n_scales, n_layers, activation,
                                         "FC_net", context=True,
                                         context_dims=self.t_dims,
                                         debug=debug)
            self.t_map = Dense(tf.reduce_prod(y_dims))
            # Context dims is what is concatenated to y. So the output of
            # t_map should have the same rank as y and the N-1 dimensions must match.

            # TODO I have to adjust flow_y in such a way that it also uses the
            # context variable
            n_scales = 2
            self.flow_z = CouplingLayers(z_dims, "flow_z", feature_maps, 0,
                                         n_scales, n_layers, activation,
                                         "FC_net", debug=debug)

            self.supervised_t = network(in_dims=x_dims, out_dims=x_dims,
                                        name_tag='sup_t', n_layers=1,
                                        feature_maps=feature_maps,
                                        squeeze=True, squeeze_dims=t_dims,
                                        debug=debug)
            self.supervised_y = network(in_dims=x_dims, out_dims=x_dims,
                                        name_tag='sup_y', n_layers=3,
                                        squeeze=True,
                                        squeeze_dims=y_dims * t_dims,
                                        debug=debug)

    def dequantize(self, z, factor=1.0):
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
        noise = tf.random.uniform(z.shape, dtype=tf.float64) * factor - 0.5 *\
            factor
        return z + noise

    @staticmethod
    def logit_normalize(z, ldj, reverse=False, alpha=1e-6):
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
            z = z / (1 - alpha) - alpha / (2 - 2 * alpha)
            z *= maximal
            ldj += tf.math.log(maximal) * tf.cast(tf.reduce_prod(z.shape[1:]),
                                                  tf.float64)

        return z, ldj

    @staticmethod
    def log_prior(z):
        """ The prior log probability of a standard Gaussian

        N(z|mu=0, Sig=1)

        Parameters
        ----------
        z : tensor
            The tensor of which the prior probability should be calculated.
            Can be any shape
        """

        pi = tf.constant(math.pi, dtype=tf.float64)
        norm_term = tf.math.log(2 * pi)
        log_p = -(norm_term + tf.math.square(z)) * 0.5
        return tf.reduce_sum(log_p, axis=tf.range(1, tf.rank(z)))

    @staticmethod
    def project(z, reverse=False):
        """ Projects z from 1D to 2D or the other way around."""
        if not reverse:
            sign = tf.math.sign(z)
            z_square = tf.math.square(z)
            z = tf.math.sqrt(0.5 * z_square) * sign
            z = tf.concat([z, z], axis=-1)
        else:
            sign = tf.math.sign(z[:, 0])
            z_square = tf.math.square(z)
            z = tf.math.sqrt(tf.reduce_sum(z_square, axis=-1, keepdims=True))
            z = z * tf.expand_dims(sign, axis=1)
        return z

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

        x = self.dequantize(x)  # TODO figure out if/when we want to dequantize
        y = self.dequantize(y, factor=1.0)
        # x, ldj = self.logit_normalize(x, ldj)
        # y, ldj = self.logit_normalize(y, ldj)

        x_intermediate, ldj = self.flow_x(x, ldj, step, training=training)
        x_intermediate = tf.reshape(x_intermediate, (len(x), -1))
        y = self.project(y)
        t = tf.reshape(self.t_map(t), (len(y), self.y_dims))

        y_intermediate, ldj = self.flow_y(y, ldj, step, training=training, t=t)
        # TODO Flow y with context t

        z = tf.concat([x_intermediate, y_intermediate], axis=-1)
        z, ldj = self.flow_z(z, ldj, step, training=training)
        log_pz = self.log_prior(z)
        log_pxy = log_pz + ldj
        log_2 = tf.math.log(tf.constant(2., dtype=tf.float64))
        bpd = -log_pxy / (tf.size(z[0], out_type=tf.float64) * log_2)


        sup_t = self.supervised_t(x, step, training=training)
        sup_y = self.supervised_y(x, step, training=training)
        sup_y = tf.reshape(sup_y, (len(x), self.y_dims, self.t_dims))
        sup_t = tf.reshape(sup_t, (len(x), 1, self.t_dims))
        sup_y = tf.reduce_sum(sup_t * sup_y, axis=-1)
        return bpd, ldj, z, sup_t, sup_y

    @tf.function
    def loss(self, features, bpd, ldj, z, sup_t, sup_y, step):
        """ Loss function of the model. """
        if self.debug:
            print("Calculating loss")
        _, _, t, _, y, *_ = features

        variational_t = self.t_loss(t, sup_t)
        variational_y = self.y_loss(y, sup_y)

        if step is not None and step % (self.log_steps * 5) == 0:
            l_step = step // (self.log_steps * 5)
            tf.summary.scalar("partial_loss/ldj",
                              tf.reduce_mean(ldj), step=l_step)
            tf.summary.scalar("partial_loss/variational_t",
                              tf.reduce_mean(variational_t), step=l_step)
            tf.summary.scalar("partial_loss/variational_y",
                              tf.reduce_mean(variational_y), step=l_step)

        loss = tf.reduce_mean(bpd + variational_t + variational_y)
        return loss

    # @tf.function
    def _reverse_flow(self, z, t, ldj):
        """ Reverse of flow"""

        z, ldj = self.flow_z(z, ldj, None, reverse=True)
        x, y = tf.split(z, self.dims_split, axis=-1)
        x = tf.reshape(x, (len(x), *self.x_dims))
        # print(y.shape, t.shape)
        y, ldj = self.flow_y(y, ldj, None, reverse=True, t=t)
        y = self.project(y, reverse=True)
        # print(y.shape)
        x, ldj = self.flow_x(x, ldj, None, reverse=True)

        # y, ldj = self.logit_normalize(y, ldj, reverse=True)
        # x, ldj = self.logit_normalize(x, ldj, reverse=True)

        return x, y

    def sample(self, n_samples):
        shape = (n_samples, self.dims[0], self.dims[1], self.dims[2] * 2)
        z = tf.random.normal(shape)
        ldj = tf.zeros(n_samples, dtype=tf.float64)
        x, y, ldj = self._reverse_flow(z, ldj, 0)
        return x, y

    def do_intervention(self, x, t, t_cf, nr_samples):
        """ Do an intervention for t=0 and t=1. """
        sup_t = self.supervised_t(x, None, training=False)
        sup_y = self.supervised_y(x, None, training=False)
        sup_y = tf.reshape(sup_y, (len(x), self.y_dims, self.t_dims))
        sup_t = tf.reshape(sup_t, (len(x), 1, self.t_dims))
        sup_y = tf.reduce_sum(sup_t * sup_y, axis=-1)

        ldj = tf.zeros(x.shape[0], dtype=tf.float64)
        x_intermediate, ldj = self.flow_x(x, ldj, None, training=False)
        x_intermediate = tf.reshape(x_intermediate, (len(x), -1))
        sup_y = self.project(sup_y)
        sup_t = tf.reshape(self.t_map(sup_t), (len(sup_y), self.y_dims))

        sup_y_intermediate, ldj = self.flow_y(sup_y, ldj, None, training=False, t=t)

        z = tf.concat([x_intermediate, sup_y_intermediate], axis=-1)
        z, ldj = self.flow_z(z, ldj, None, training=False)

        x_pred, y_pred = self._reverse_flow(z, t, ldj)
        x_pred_cf, y_pred_cf = self._reverse_flow(z, t_cf, ldj)
        return y_pred, y_pred_cf


def test_model():
    ds = tfds.load('cifar10')

    # dims = (32, 32, 3)
    x_dims = 102
    y_dims = 1
    t_dims = 20
    name_tag = "test"
    feature_maps = 32
    n_scales = 2
    n_layers = 4
    activation = "relu"
    # architecture_type = "ResNet"
    architecture_type = "FC_net"
    batch_size = 8

    # for data in ds['train'].batch(2 * batch_size).take(1):
    #     images = data['image']
    #     x = images[:batch_size]
    #     y = images[batch_size:]

    x = tf.ones((batch_size, x_dims), dtype=tf.float64)
    y = tf.ones((batch_size, y_dims), dtype=tf.float64)
    t = tf.ones((batch_size, t_dims), dtype=tf.float64)

    # TODO writ test for logit normalise
    start_time = time.time()
    model = CRNVP(x_dims, y_dims, t_dims, name_tag, feature_maps,
                          n_scales, n_layers,
                          activation, architecture_type, debug=True)
    middle_time = time.time()
    bpd, ldj, z = model(x, t, y, 0, training=True)
    end_time = time.time()

    x_recon, y_recon, _ = model._reverse_flow(z, t, bpd)
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
