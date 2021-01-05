import math
import tensorflow as tf
import time

from coupling import CouplingLayers
from fc_net import FC_net
from res_net import ResNet
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten


class NCF(Model):
    """ Combination model of several normalising flow and causality models.

    Contains one part that infers z from x, by mapping x to a Gaussian prior.
    Then learns the mapping from y to its prior, while using the inferred z
    and t as context in that flow.

    Inference in training time is done by inferring z from x, and then mapping
    the prior of y to y with z and the intervention variable. We can sample the
    prior on y multiple times to get a more accurate estimate.
    """

    def __init__(
        self,
        x_dims=(50, 50, 3),
        t_dims=2,
        y_dims=1,
        z_dims=(50, 50, 3),
        name_tag="normalising_causal_flow",
        feature_maps=32,
        n_scales=3,
        n_layers=3,
        activation="elu",
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
        self.x_dims = x_dims
        self.t_dims = t_dims
        self.y_dims = y_dims
        self.z_dims = z_dims
        network = eval(architecture_type)

        self.annealing_factor = 1e-8
        self.log_2 = tf.math.log(tf.constant(2., dtype=tf.float64))

        self.flatten = Flatten()
        if architecture_type == "ResNet":
            self.z_proj_dims = 512 - t_dims - y_dims
        else:
            self.z_proj_dims = 45
        tz_dims = int(self.z_proj_dims + t_dims)
        self.z_proj = Dense(self.z_proj_dims)
        self.z_proj.build((None, tf.reduce_prod(x_dims)))

        if x_dims == int and x_dims % 4 != 0:
            n_scales = 2

        n_scales = 2
        self.flow_xz = CouplingLayers(x_dims,
                                      "flow_xz",
                                      feature_maps,
                                      0,
                                      n_scales,
                                      n_layers=n_layers,
                                      activation=activation,
                                      architecture_type=architecture_type,
                                      debug=debug)
        self.flow_y = CouplingLayers(y_dims,
                                     "flow_y",
                                     feature_maps,
                                     0,
                                     1,
                                     n_layers=n_layers,
                                     activation=activation,
                                     architecture_type="FC_net",
                                     context=True,
                                     context_dims=tz_dims,
                                     debug=debug)

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
        noise = tf.random.uniform(z.shape, dtype=tf.float64) - 0.5
        return z + noise

    def log_prior(self, z):
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

    def call(self, x, t, y, step, training=False):
        """ Infers the latent confounder and the log-likelihood of the input.

        Calculates the latent confouder z by mapping the input through several
        coupling layers.
        //It also calculates the log-likelihood of the input
        //triplet by adding the log-likelihood of z to the log determinant
        //Jacobian of the mapping.

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
        """

        ldj_xz = tf.zeros(x.shape[0], dtype=tf.float64)
        ldj_y = tf.zeros(y.shape[0], dtype=tf.float64)
        x = self.dequantize(x)

        z, ldj_xz = self.flow_xz(x, ldj_xz, step, training=training)
        log_pz = self.log_prior(z)
        log_px = log_pz + ldj_xz
        bpd_z = -log_px / (tf.size(z[0], out_type=tf.float64) * self.log_2)

        f_z = self.flatten(z)
        projected_z = self.z_proj(f_z)
        context = tf.concat([projected_z, t], 1)
        y_prior, ldj_y = self.flow_y(y, ldj_y, step,
                                     training=training, t=context)
        log_py = self.log_prior(y_prior) + ldj_y
        bpd_y = -log_py / (tf.size(y[0], out_type=tf.float64) * self.log_2)
        return bpd_z, bpd_y, z

    @tf.function
    def loss(self, features, bpd_z, bpd_y, z, step):
        """ Loss function of the model. """
        if self.debug:
            print("Calculating loss")

        if step is not None and step % (self.log_steps * 10) == 0:
            l_step = step // (self.log_steps * 10)
            tf.summary.scalar("partial_loss/bpd_z",
                              tf.reduce_mean(bpd_z), step=l_step)
            tf.summary.scalar("partial_loss/bpd_y",
                              tf.reduce_mean(bpd_y), step=l_step)

        loss = tf.reduce_mean(bpd_z + bpd_y)
        return loss

    def do_intervention(self, x, t0, t1, n_samples):
        ldj = tf.zeros(x.shape[0], dtype=tf.float64)
        x = self.dequantize(x)
        z, ldj_xz = self.flow_xz(x, ldj, None, training=False)
        ldj = tf.zeros(x.shape[0] * n_samples, dtype=tf.float64)
        f_z = self.flatten(z)
        projected_z = self.z_proj(f_z)
        projected_z = tf.tile(projected_z, [n_samples, 1])

        t0 = tf.tile(t0, [n_samples, 1])
        t1 = tf.tile(t1, [n_samples, 1])
        y_prior = tf.random.normal((z.shape[0] * n_samples, self.y_dims),
                                   dtype=tf.float64)

        context = tf.concat([projected_z, t0], -1)
        y0, ldj_y = self.flow_y(y_prior, ldj, None, reverse=True,
                                training=False, t=context)
        y0 = tf.reshape(y0, (x.shape[0], -1, 1))
        y0 = tf.reduce_mean(y0, 1)

        context = tf.concat([projected_z, t1], -1)
        y1, ldj_y = self.flow_y(y_prior, ldj, None, reverse=True,
                                training=False, t=context)
        y1 = tf.reshape(y1, (x.shape[0], -1, 1))
        y1 = tf.reduce_mean(y1, 1)
        return y0, y1
