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
        y_dims=1,
        z_dims=32,
        category_sizes=2,
        t_loss=CategoricalCrossentropy(),
        y_loss=MeanSquaredError(),
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
        self.t_loss = t_loss
        self.y_dims = y_dims
        self.y_loss = y_loss
        self.log_steps = log_steps
        self.architecture_type = architecture_type

        if architecture_type == "ResNet":
            self.x_cat_dims = x_cat_dims
            x_dims = x_cont_dims[:-1] + (x_cont_dims[-1] + x_cat_dims[-1] *
                                         category_sizes, )
        else:
            self.x_cat_dims = (x_cat_dims, )
            x_dims = x_cont_dims + x_cat_dims * category_sizes

        self.encode = Encoder(x_dims, t_dims, y_dims, z_dims, "Encoder",
                              feature_maps, architecture_type, debug)
        self.decode = Decoder(x_cat_dims, x_cont_dims, t_dims, y_dims, z_dims,
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
        x_cat, x_cont, t, y, *_ = features
        qt_prob, qy_mean, qz_mean, qz_std = encoder_params
        x_cat_prob, x_cont_mean, x_cont_std, t_prob, y_mean = decoder_params
        x_cat = tf.reshape(x_cat, (len(x_cat), *self.x_cat_dims,
                                   self.category_sizes))

        distortion_x = CategoricalCrossentropy()(x_cat, x_cat_prob) \
            - get_log_prob(x_cont, 'N', mean=x_cont_mean,
                           std=x_cont_std)
        distortion_t = self.t_loss(t, t_prob)
        distortion_y = self.y_loss(y, y_mean)

        rate = get_analytical_KL_divergence(qz_mean, qz_std)

        variational_t = self.t_loss(t, qt_prob)
        variational_y = self.y_loss(y, qy_mean)

        if step is not None and step % (self.log_steps * 5) == 0:
            l_step = step // (self.log_steps * 5)
            if self.architecture_type == "ResNet":
                with tf.device("cpu:0"):
                    tf.summary.image("reconstructed_proxy",
                                     x_cont_mean, step=l_step, max_outputs=4)
            tf.summary.scalar("partial_loss/distortion_x",
                              tf.reduce_mean(distortion_x), step=l_step)
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

        elbo_local = -(rate + distortion_x + distortion_t + distortion_y +
                       variational_t + variational_y)
        elbo = tf.reduce_mean(elbo_local)
        return -elbo

    def do_intervention(self, x, nr_samples):
        *_, qz_mean, qz_std = self.encode(x, None, None, None, training=False)
        final_shape = (nr_samples, qz_mean.shape[0], self.y_dims, self.t_dims)
        qz = tf.random.normal((nr_samples, *qz_mean.shape), dtype=tf.float64)
        z = qz * qz_std + qz_mean
        y = self.decode.do_intervention(z, nr_samples)
        y_mean = tf.reduce_mean(tf.reshape(y, final_shape), axis=0)
        mu_y0, mu_y1 = y_mean[..., 0], y_mean[..., 1]
        return mu_y0, mu_y1


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
        y_dims=1,
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
        self.y_dims = y_dims
        self.z_dims = z_dims
        self.debug = debug
        network = eval(architecture_type)

        if architecture_type == "ResNet":
            intermediate_dims = feature_maps * 16
        else:
            intermediate_dims = feature_maps

        self.qt_logits = network(in_dims=x_dims, out_dims=x_dims,
                                 name_tag="qt", n_layers=1,
                                 feature_maps=feature_maps * 8,
                                 squeeze=True, squeeze_dims=t_dims,
                                 debug=debug)

        self.hqy = network(in_dims=x_dims, out_dims=x_dims,
                           name_tag="hqy", n_layers=3,
                           feature_maps=feature_maps, squeeze=True,
                           squeeze_dims=intermediate_dims, debug=debug)

        self.mu_qy_t = FC_net(in_dims=intermediate_dims, out_dims=y_dims * t_dims,
                               name_tag="mu_qy_t", n_layers=2,
                               feature_maps=feature_maps, debug=debug)

        self.hqz = FC_net(in_dims=intermediate_dims + y_dims,
                          out_dims=intermediate_dims,
                          name_tag="hqz", n_layers=2,
                          feature_maps=feature_maps, debug=debug)

        self.qz_t = FC_net(in_dims=intermediate_dims, out_dims=z_dims * 2 * t_dims,
                           name_tag="qz_t", n_layers=2,
                           feature_maps=feature_maps, debug=debug)

    @tf.function
    def call(self, x, t, y, step, training=False):
        if self.debug:
            print("Encoding")
        qt_prob = nn.softmax(self.qt_logits(x, step, training=training))
        qt = tfd.Independent(tfd.OneHotCategorical(probs=qt_prob,
                                                   dtype=tf.float64),
                             reinterpreted_batch_ndims=1,
                             name="qt")
        qt_sample = qt.sample()

        hqy = self.hqy(x, step, training=training)
        mu_qy_t = self.mu_qy_t(hqy, step, training=training)
        mu_qy_t = tf.reshape(mu_qy_t, (len(x), self.y_dims, self.t_dims))
        shape = tf.concat([[qt_sample.shape[0]],
                           tf.ones(tf.rank(mu_qy_t) - 2, dtype=tf.int32),
                           [qt_sample.shape[-1]]], axis=0)
        if training:
            t = tf.reshape(t, shape)
            qy_mean = tf.reduce_sum(t * mu_qy_t, axis=-1)
        else:
            qt_sample = tf.reshape(qt_sample, shape)
            qy_mean = tf.reduce_sum(qt_sample * mu_qy_t, axis=-1)
        qy = tfd.Independent(tfd.Normal(loc=qy_mean,
                                        scale=tf.ones_like(qy_mean)),
                             reinterpreted_batch_ndims=1,
                             name="qy")

        if training:
            xy = tf.concat([hqy, y], -1)
        else:
            xy = tf.concat([hqy, qy.sample()], -1)

        hidden_z = self.hqz(xy, step, training=training)
        mu_qz_t = self.qz_t(hidden_z, step, training=training)
        mu_qz_t = tf.reshape(mu_qz_t, (len(x), self.z_dims * 2, self.t_dims))
        qz_mean, qz_std = tf.split(mu_qz_t, 2, axis=-2)

        if training:
            qz_mean = tf.reduce_sum(t * qz_mean, axis=-1)
            qz_std = tf.reduce_sum(t * softplus(qz_std), axis=-1)
        else:
            qz_mean = tf.reduce_sum(qt_sample * qz_mean, axis=-1)
            qz_std = tf.reduce_sum(qt_sample * softplus(qz_std), axis=-1)
        return qt_prob, qy_mean, qz_mean, qz_std


class Decoder(Model):

    def __init__(
        self,
        x_cat_dims=10,
        x_cont_dims=10,
        t_dims=2,
        y_dims=1,
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
        self.y_dims = y_dims
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
                                debug=debug)
        self.t_logits = FC_net(in_dims=z_dims, out_dims=z_dims, name_tag="t",
                               n_layers=1, feature_maps=feature_maps,
                               squeeze=True, squeeze_dims=t_dims,
                               debug=self.debug)

        self.mu_y_t = FC_net(in_dims=z_dims, out_dims=y_dims * t_dims,
                             name_tag="mu_y_t", n_layers=2,
                             feature_maps=feature_maps, debug=debug)

    @tf.function
    def call(self, z, t, step, training=False):
        if self.debug:
            print("Decoding")
        x_logits = self.x_logits(z, step, training=training)
        x_cont_mean, x_cont_std, x_cat_logits = tf.split(x_logits,
                                                         self.x_split_dims,
                                                         axis=-1)
        x_cont_std = softplus(x_cont_std)
        x_cat_prob = nn.softmax(tf.reshape(x_cat_logits, (len(z),
                                                          *self.x_cat_dims,
                                                          self.category_sizes
                                                          )))

        t_prob = nn.softmax(self.t_logits(z, step, training=training))
        mu_y_t = self.mu_y_t(z, step, training=training)
        mu_y_t = tf.reshape(mu_y_t, (len(z), self.y_dims, self.t_dims))

        shape = tf.concat([[t.shape[0]],
                           tf.ones(tf.rank(mu_y_t) - 2, dtype=tf.int32),
                           [t.shape[-1]]], axis=0)
        t = tf.reshape(t, shape)
        y_mean = tf.reduce_sum(t * mu_y_t, axis=-1)

        return x_cat_prob, x_cont_mean, x_cont_std, t_prob, y_mean

    def do_intervention(self, z, nr_samples):
        """ Computes the quantity E[y|z, do(t=0)] and E[y|z, do(t=1)]

    `   nr_sample of samples are drawn from the Normal distribution
        N(qz_mean, qz_std) and used to infer y for both t=0 and t=1.
        The samples are averaged at the end.

        """
        # Is this correct? Do we average and sample correctly?

        y = self.mu_y_t(z, None, training=False)
        # mean_y = tf.reduce_mean(y, axis=0)
        # return mean_y[..., 0], mean_y[..., 1]
        return y
