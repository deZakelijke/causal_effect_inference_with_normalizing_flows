import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.activations import softplus
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow import nn
from tensorflow_probability import distributions as tfd

from fc_net import FC_net
from res_net import ResNet
from utils import get_log_prob, get_analytical_KL_divergence


class CEVAE(Model):
    """ CEVAE model with fc nets between random variables.
    https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py

    After some attempts to port the example code 1 to 1 to TF2, I decided
    to restructure the encoder and decoder to a Model subclass instead to
    two functions.

    Several fc_nets have an output size of *2 something. This is to output
    both the mean and std of a Normal distribution at once.
    """

    def __init__(
        self,
        x_dims=30,
        x_cat_dims=10,
        x_cont_dims=10,
        t_dims=2,
        t_type='Categorical',
        y_dims=1,
        y_type='Normal',
        z_dims=32,
        category_sizes=2,
        name_tag="no_name",
        feature_maps=256,
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
        self.category_sizes = category_sizes
        self.t_dims = t_dims
        self.y_dims = y_dims
        self.log_steps = log_steps
        self.architecture_type = architecture_type

        # self.annealing_factor = 1e-4
        self.annealing_factor = 1.

        if architecture_type == "ResNet":
            self.x_cat_dims = x_cat_dims
        else:
            self.x_cat_dims = (x_cat_dims, )

        if t_type == "Categorical":
            self.t_loss = CategoricalCrossentropy()
            t_dist = lambda x: tfd.OneHotCategorical(probs=x,
                                                     dtype=tf.float64)
            t_activation = nn.softmax
        else:
            self.t_loss = MeanSquaredError()
            t_dist = lambda x: tfd.Normal(x, scale=tf.ones_like(x))
            t_activation = lambda x: x

        if y_type == "Categorical":
            self.y_loss = CategoricalCrossentropy()
            y_dist = lambda x: tfd.OneHotCategorical(probs=x,
                                                     dtype=tf.float64)
            y_activation = nn.softmax
        else:
            self.y_loss = MeanSquaredError()
            y_dist = lambda x: tfd.Normal(x, scale=tf.ones_like(x))
            y_activation = lambda x: x

        self.encode = Encoder(x_dims, t_dims, t_dist, t_activation,
                              y_dims, y_dist, y_activation, z_dims,
                              "Encoder", feature_maps, architecture_type,
                              debug)
        self.decode = Decoder(x_cat_dims, x_cont_dims, t_dims,
                              t_activation, y_dims, y_activation, z_dims,
                              category_sizes, "Decoder", feature_maps,
                              architecture_type, debug)

    @tf.function
    def call(self, x, t, y, step, training=False):
        """ Forward pass of the CEVAE

        Args:   features, tuple of all input variables
                step, iteration number of the model for tensorboard logging

        Returns:    encoder_params, parameters of all variational distributions
                    qt_prob, qy_mean, qz_mean, qz_std

                    decoder_params, parameters of all likelihoods
                    x_cat_prob, x_cont_mean, x_cont_std, t_prob, y_mean
        """
        if self.debug:
            print("Starting forward pass")

        encoder_params = self.encode(x, t, y, step, training=training)
        _, _, qz_mean, qz_std = encoder_params
        qz = tf.random.normal(qz_mean.shape, dtype=tf.float64)
        qz = qz * qz_std + qz_mean

        decoder_params = self.decode(qz, t, step, training=training)
        return encoder_params, decoder_params

    @tf.function
    def loss(self, features, encoder_params, decoder_params, step):
        if self.debug:
            print("Calculating loss")
        x_cat, x_cont, t, _, y, *_ = features
        qt_prob, qy_mean, qz_mean, qz_std = encoder_params
        x_cat_prob, x_cont_mean, x_cont_std, t_prob, y_mean = decoder_params
        x_cat = tf.reshape(x_cat, (len(x_cat), *self.x_cat_dims,
                                   self.category_sizes))

        distortion_x_cat = CategoricalCrossentropy()(x_cat, x_cat_prob)
        # Use relu to set nans to zero
        distortion_x_cont = nn.relu(MeanSquaredError()(x_cont, x_cont_mean))
        distortion_x = distortion_x_cat + distortion_x_cont
        distortion_t = self.t_loss(t, t_prob)
        distortion_y = self.y_loss(y, y_mean)

        rate = get_analytical_KL_divergence(qz_mean, qz_std)

        variational_t = self.t_loss(t, qt_prob)
        variational_y = self.y_loss(y, qy_mean)

        if step is not None and step % (self.log_steps * 10) == 0:
            l_step = step // (self.log_steps * 10)
            if self.architecture_type == "ResNet":
                with tf.device("cpu:0"):
                    tf.summary.image("reconstructed_proxy",
                                     x_cont_mean, step=l_step, max_outputs=4)
            tf.summary.scalar("partial_loss/distortion_x",
                              tf.reduce_mean(distortion_x), step=l_step)
            tf.summary.scalar("partial_loss/distortion_x_cat",
                              tf.reduce_mean(distortion_x_cat), step=l_step)
            tf.summary.scalar("partial_loss/distortion_x_cont",
                              tf.reduce_mean(distortion_x_cont), step=l_step)

            tf.summary.scalar("partial_loss/distortion_t",
                              tf.reduce_mean(distortion_t), step=l_step)
            tf.summary.scalar("partial_loss/distortion_y",
                              tf.reduce_mean(distortion_y), step=l_step)
            tf.summary.scalar("partial_loss/rate_z",
                              tf.reduce_mean(rate), step=l_step)
            tf.summary.scalar("partial_loss/variational_t",
                              tf.reduce_mean(variational_t), step=l_step)
            tf.summary.scalar("partial_loss/variational_y",
                              tf.reduce_mean(variational_y), step=l_step)

        elbo_local = -(self.annealing_factor * rate +
                       distortion_x +
                       distortion_t +
                       distortion_y +
                       variational_t +
                       variational_y
                       )
        elbo = tf.reduce_mean(elbo_local)
        return -elbo

    def do_intervention(self, x, t0, t1, n_samples):
        """ Perform two interventions to compare downstream.

        Use n_samples for both number of samples from latent space
        and for number of samples from intervention distribution.

        """
        # Get latent confounder
        *_, qz_mean, qz_std = self.encode(x, None, None, None, training=False)
        # final_shape = (n_samples, qz_mean.shape[0], self.y_dims, self.t_dims)
        qz = tf.random.normal((n_samples, *qz_mean.shape), dtype=tf.float64)
        z = qz * qz_std + qz_mean

        # Do simulation with intervention variable
        # We have to sample t0 and t1 several times
        y0, y1 = self.decode.do_intervention(z, t0, t1, n_samples)
        return y0, y1


class Encoder(Model):
    """
    New plan, we assume y is a scalar or vector to make life a bit simpler.
    That means that networks generating y have to have a bottleneck at the end.
    We can then also assume that z is a vector, so networks generating z will
    have to have a bottleneck as well. From z to x we might need a reshape
    or upsampling thingy.
    """

    def __init__(
        self,
        x_dims=30,
        t_dims=2,
        t_dist=None,
        t_activation=None,
        y_dims=1,
        y_dist=None,
        y_activation=None,
        z_dims=32,
        name_tag="no_name",
        feature_maps=200,
        architecture_type="FC_net",
        debug=False
    ):
        """
        Parameters
        ----------

        """

        super().__init__(name=name_tag)
        self.t_dims = t_dims
        self.t_dist = t_dist
        self.t_activation = t_activation
        self.y_dims = y_dims
        self.y_dist = y_dist
        self.y_activation = y_activation
        self.z_dims = z_dims
        self.debug = debug
        network = eval(architecture_type)

        if architecture_type == "ResNet":
            intermediate_dims = feature_maps * 2
        else:
            intermediate_dims = feature_maps // 2

        self.qt_logits = network(in_dims=x_dims, out_dims=x_dims,
                                 name_tag="qt", n_layers=1,
                                 feature_maps=feature_maps * 8,
                                 squeeze=True, squeeze_dims=t_dims,
                                 debug=debug)

        self.hqy = network(in_dims=x_dims, out_dims=x_dims,
                           name_tag="hqy", n_layers=3,
                           feature_maps=feature_maps, squeeze=True,
                           squeeze_dims=intermediate_dims, debug=debug)

        self.mu_qy_t = FC_net(in_dims=intermediate_dims + t_dims,
                              out_dims=y_dims,
                              name_tag="mu_qy_t", n_layers=3,
                              feature_maps=feature_maps, debug=debug)
        # Make input shape x + t, output shape y, more layers

        # self.hqz = FC_net(in_dims=intermediate_dims + y_dims,
        #                   out_dims=intermediate_dims,
        #                   name_tag="hqz", n_layers=2,
        #                   feature_maps=feature_maps, debug=debug)
        # Drop this network?

        self.qz_t = FC_net(in_dims=intermediate_dims + y_dims + t_dims,
                           out_dims=z_dims * 2,
                           name_tag="qz_t", n_layers=4,
                           feature_maps=feature_maps, debug=debug)

    @tf.function
    def call(self, x, t, y, step, training=False):
        if self.debug:
            print("Encoding")
        qt_prob = self.t_activation(self.qt_logits(x, step, training=training))
        qt = tfd.Independent(self.t_dist(qt_prob),
                             reinterpreted_batch_ndims=1,
                             name="qt")
        qt_sample = qt.sample()

        hqy = self.hqy(x, step, training=training)
        # mu_qy_t = self.mu_qy_t(hqy, step, training=training)
        # mu_qy_t = tf.reshape(mu_qy_t, (len(x), self.y_dims, self.t_dims))
        # shape = tf.concat([[qt_sample.shape[0]],
        #                    tf.ones(tf.rank(mu_qy_t) - 2, dtype=tf.int32),
        #                    [qt_sample.shape[-1]]], axis=0)
        if training:
            # t = tf.reshape(t, shape)
            # qy_mean = tf.reduce_sum(t * mu_qy_t, axis=-1)
            qy_mean = self.mu_qy_t(tf.concat([hqy, t], -1), step,
                                   training=training)
        else:
            # qt_sample = tf.reshape(qt_sample, shape)
            # qy_mean = tf.reduce_sum(qt_sample * mu_qy_t, axis=-1)
            qy_mean = self.mu_qy_t(tf.concat([hqy, qt_sample], -1), step,
                                   training=training)
        qy_mean = self.y_activation(qy_mean)
        qy = tfd.Independent(self.y_dist(qy_mean),
                             reinterpreted_batch_ndims=1,
                             name="qy")

        if training:
            xyt = tf.concat([hqy, y, t], -1)
        else:
            xyt = tf.concat([hqy, qy.sample(), qt_sample], -1)

        # hidden_z = self.hqz(xy, step, training=training)
        # mu_qz_t = self.qz_t(hidden_z, step, training=training)
        # mu_qz_t = tf.reshape(mu_qz_t, (len(x), self.z_dims * 2, self.t_dims))
        mu_qz_t = self.qz_t(xyt, step, training=training)
        qz_mean, qz_std = tf.split(mu_qz_t, 2, axis=-1)
        qz_std = softplus(qz_std)

        # if training:
        #     qz_mean = tf.reduce_sum(t * qz_mean, axis=-1)
        #     qz_std = tf.reduce_sum(t * softplus(qz_std), axis=-1)
        # else:
        #     qz_mean = tf.reduce_sum(qt_sample * qz_mean, axis=-1)
        #     qz_std = tf.reduce_sum(qt_sample * softplus(qz_std), axis=-1)
        return qt_prob, qy_mean, qz_mean, qz_std


class Decoder(Model):

    def __init__(
        self,
        x_cat_dims=10,
        x_cont_dims=10,
        t_dims=2,
        t_activation=None,
        y_dims=1,
        y_activation=None,
        z_dims=32,
        category_sizes=2,
        name_tag="no_name",
        feature_maps=256,
        architecture_type="FC_net",
        debug=False,
        **_
    ):
        """
        Parameters
        ----------

        """

        super().__init__(name=name_tag)
        self.category_sizes = category_sizes
        self.t_dims = t_dims
        self.t_activation = t_activation
        self.y_dims = y_dims
        self.y_activation = y_activation
        self.z_dims = z_dims
        self.debug = debug
        network = eval(architecture_type)

        if architecture_type == "ResNet":
            x_out_dims = x_cont_dims[:-1] + (x_cont_dims[-1] * 2 +
                                             x_cat_dims[-1] * category_sizes, )
            self.x_split_dims = [x_cont_dims[-1], x_cont_dims[-1],
                                 x_cat_dims[-1] * category_sizes]
            self.x_cat_dims = x_cat_dims
            intermediate_dims = x_cont_dims[:-1]
        else:
            x_out_dims = x_cont_dims * 2 + x_cat_dims * category_sizes
            self.x_split_dims = [x_cont_dims, x_cont_dims,
                                 x_cat_dims * category_sizes]
            self.x_cat_dims = (x_cat_dims, )
            intermediate_dims = x_cont_dims

        self.x_logits = network(in_dims=z_dims, out_dims=x_out_dims,
                                name_tag="x", n_layers=3,
                                feature_maps=feature_maps,
                                unsqueeze=True, squeeze_dims=intermediate_dims,
                                coord_conv=False, debug=debug)
        self.t_logits = FC_net(in_dims=z_dims, out_dims=z_dims, name_tag="t",
                               n_layers=1, feature_maps=feature_maps * 4,
                               squeeze=True, squeeze_dims=t_dims,
                               debug=self.debug)

        self.mu_y_t = FC_net(in_dims=z_dims + t_dims, out_dims=y_dims,
                             name_tag="mu_y_t", n_layers=3,
                             feature_maps=feature_maps, debug=debug)

    @tf.function
    def call(self, z, t, step, training=False):
        if self.debug:
            print("Decoding")
        x_logits = self.x_logits(z, step, training=training)
        x_cont_mean, x_cont_logvar, x_cat_logits = tf.split(x_logits,
                                                            self.x_split_dims,
                                                            axis=-1)
        x_cont_std = softplus(x_cont_logvar)
        x_cat_prob = nn.softmax(tf.reshape(x_cat_logits, (len(z),
                                                          *self.x_cat_dims,
                                                          self.category_sizes
                                                          )))

        t_prob = self.t_activation(self.t_logits(z, step, training=training))
        # mu_y_t = self.mu_y_t(z, step, training=training)
        # mu_y_t = tf.reshape(mu_y_t, (len(z), self.y_dims, self.t_dims))
        #
        # shape = tf.concat([[t.shape[0]],
        #                    tf.ones(tf.rank(mu_y_t) - 2, dtype=tf.int32),
        #                    [t.shape[-1]]], axis=0)
        # t = tf.reshape(t, shape)
        # y_mean = self.y_activation(tf.reduce_sum(t * mu_y_t, axis=-1))
        y_mean = self.mu_y_t(tf.concat([z, t], -1), step, training=training)
        y_mean = self.y_activation(y_mean)

        return x_cat_prob, x_cont_mean, x_cont_std, t_prob, y_mean

    def do_intervention(self, z, t0, t1, n_samples):
        """ Computes the quantity E[y|z, do(t=0)] and E[y|z, do(t=1)]

    `   nr_sample of samples are drawn from the Normal distribution
        N(qz_mean, qz_std) and used to infer y for both t=0 and t=1.
        The samples are averaged at the end.

        """
        # Is this correct? Do we average and sample correctly?
        in0 = tf.concat([z, tf.tile(tf.expand_dims(t0, 0),
                                    [n_samples, 1, 1])], -1)
        in1 = tf.concat([z, tf.tile(tf.expand_dims(t1, 0),
                                    [n_samples, 1, 1])], -1)
        y0 = self.mu_y_t(in0, None, training=False)
        y1 = self.mu_y_t(in1, None, training=False)

        y0 = tf.reduce_mean(y0, axis=0)
        y1 = tf.reduce_mean(y1, axis=0)
        return y0, y1
