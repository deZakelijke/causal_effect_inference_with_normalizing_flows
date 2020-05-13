import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.activations import softplus
from tensorflow import nn
from tensorflow_probability import distributions as tfd

from fc_net import FC_net
from res_net import ResNet
from utils import get_log_prob, get_analytical_KL_divergence

# Add extra output to make t one-hot encoded
"""
    So we have the situation that t is not always binary. In that case
    we would like to one-hot encode it because it isn't ordinal either.
    But now we have a dimensionality problem. In the binary case we have
    one scalar t per data sample. We can broadcast that over the entire shape
    of the sample and multiply to get the appropriate value. In the one-hot
    case we have a vector t with one 1. You can't directly do an elementwise
    multiplication now. This is because the dimensionality of the one-hot
    vector is probably different than the one of our data sample.
    Perhaps we can stick to our initial approach. We would have a stack of
    logits from our stack of networks anyway because we have a mixture type
    setup. So perhaps we can multiply this stack of samples with our one-hot
    vector. After that we reduce over the last dimension, the dimensionality
    of t, and we have our output. But this would mean that t always has to be
    multinomial, not binary, and we always have to pass the dimensionality of
    t along from the dataset to the model.
    Does this pose problems for the crnvp?

"""


class CEVAE(Model):

    def __init__(self, x_cat_dims=10, x_cont_dims=10, t_dims=2, y_dims=1,
                 z_dims=32, category_sizes=2, name_tag="no_name",
                 feature_maps=256, architecture_type="FC_net", debug=False):
        """ CEVAE model with fc nets between random variables.
        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py

        After some attempts to port the example code 1 to 1 to TF2, I decided
        to restructure the encoder and decoder to a Model subclass instead to
        two functions.

        Several fc_nets have an output size of *2 something. This is to output
        both the mean and std of a Normal distribution at once.
        """
        super().__init__(name=name_tag)
        self.debug = debug
        self.category_sizes = category_sizes

        self.encode = Encoder(x_dims, t_dims, y_dims, z_dims, name_tag,
                              feature_maps, architecture_type, debug)
        self.decode = Decoder(x_cat_dims, x_cont_dims, t_dims, y_dims, z_dims,
                             category_sizes, name_tag, feature_maps,
                             architecture_type, debug)

        if params['dataset'] == "SHAPES":
            self.x_cat_size = params['x_cat_size']
        else:
            self.x_cat_size = (params['x_cat_size'], )

    @tf.function
    def call(self, features, step, training=False):
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
        x_cat, x_cont, t, y, y_cf, mu_0, mu_1 = features
        x = tf.concat([x_cat, x_cont], -1)

        encoder_params = self.encode(x, t, y, step, training=training)
        _, _, qz_mean, qz_std = encoder_params
        qz = tf.random.normal(qz_mean.shape, dtype=tf.float64)
        qz = qz * qz_std + qz_mean

        decoder_params = self.decode(qz, t, step, training=training)
        return encoder_params, decoder_params

    # @tf.function
    def loss(self, features, encoder_params, decoder_params, step, params):
        if self.debug:
            print("Calculating loss")
        x_cat, x_cont, t, y, y_cf, mu_0, mu_1 = features
        qt_prob, qy_mean, qz_mean, qz_std = encoder_params
        # x_cat_prob, x_cont_mean, x_cont_std, t_prob, y_mean = decoder_params
        x_cont_mean, x_cont_std, t_prob, y_mean = decoder_params

        batch_size, *features = x_cat.shape
        new_shape = (batch_size, *self.x_cat_size, self.category_sizes)
        # x_cat_prob = tf.reshape(x_cat_prob, new_shape)
        x_cat = tf.reshape(x_cat, new_shape)

        y_type = params['dataset_distributions']['y']
        t_type = params['dataset_distributions']['t']

        # distortion_x = -get_log_prob(x_cat, 'M', probs=x_cat_prob) \
        #                - get_log_prob(x_cont, 'N', mean=x_cont_mean,
        #                               std=x_cont_std)
        # print(x_cont.shape)
        # print(x_cont_mean.shape)
        # distortion_x = -get_log_prob(x_cat, 'M', probs=x_cat_prob)
        distortion_x = -get_log_prob(x_cont, 'N', mean=x_cont_mean,
                                     std=x_cont_std)
        # print(distortion_x)

        distortion_t = -get_log_prob(t, t_type, probs=t_prob)
        distortion_y = -get_log_prob(y, y_type, mean=y_mean, probs=y_mean)

        rate = get_analytical_KL_divergence(qz_mean, qz_std)

        variational_t = -get_log_prob(t, t_type, probs=qt_prob)
        variational_y = -get_log_prob(y, y_type, mean=qy_mean)

        if step is not None and step % (params['log_steps'] * 5) == 0:
            l_step = step // (params['log_steps'] * 5)
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

        print(tf.reduce_mean(distortion_x))
        print(tf.reduce_mean(distortion_t))
        print(tf.reduce_mean(distortion_y))
        print(tf.reduce_mean(rate))
        print(tf.reduce_mean(variational_t))
        print(tf.reduce_mean(variational_y))
        elbo_local = -(rate + distortion_x + distortion_t + distortion_y +
                       variational_t + variational_y)
        elbo = tf.reduce_mean(elbo_local)
        return -elbo

    def do_intervention(self, x, nr_samples):
        _, _, qz_mean, qz_std = self.encode(x, None, None, None,
                                            training=False)

        qz = tf.random.normal((nr_samples, *qz_mean.shape), dtype=tf.float64)
        z = qz * qz_std + qz_mean

        mu_y0, mu_y1 = self.decode.do_intervention(z, nr_samples)
        return mu_y0, mu_y1


class Encoder(Model):

    def __init__(self, x_dims=30, t_dims=2, y_dims=1, z_dims=32,
                 name_tag="no_name", feature_maps=200,
                 architecture_type="FC_net", debug=False):
        """
        Parameters
        ----------

        """

        super().__init__(name=name_tag)
        # self.x_cat_size = params["x_cat_size"]
        # x_cont_size = params["x_cont_size"]
        # z_size = params["z_size"]
        # y_size = params["y_size"]
        self.t_dims = t_dims
        self.debug = debug

        if architecture_type == "ResNet":
            intermediate_dims = x_dims
            y_out_dims = y_dims[:-1] + (y_dims[-1] * t_dims, )
            z_in_dims = x_dims[:-1] + (x_dims[-1] + y_dims[-1], )
            z_net_out_dims = z_dims[:-1] + (z_dims[-1] * 2 * t_dims, )
            self.y_dims = y_dims
            self.z_dims = z_dims[:-1] + (z_dims[-1] * 2, )
        else:
            intermediate_dims = feature_maps
            y_out_dims = y_dims * t_dims
            z_in_dims = x_dims + y_dims
            z_net_out_dims = z_dims * 2 * t_dims
            self.y_dims = (y_dims, )
            self.z_dims = (z_dims * 2, )
        network = eval(architecture_type)

        self.qt_logits = network(in_dims=x_dims, out_dims=dims_x,
                                 name_tag="qt", nr_layers=1,
                                 feature_maps=feature_maps,
                                 squeeze=True, squeeze_dims=t_dims,
                                 debug=debug)

        self.hqy = network(in_dims=x_dims, out_dims=intermediate_dims,
                           name_tag="hqy", nr_layers=2,
                           feature_maps=feature_maps, debug=debug)

        self.mu_qy_t = network(in_dims=intermediate_dims, out_dims=y_out_dims,
                               name_tag="mu_qy_t", nr_layers=2,
                               feature_maps=feature_maps * 2, debug=debug)

        self.hqz = network(in_dims=z_in_dims, out_dims=intermediate_dims,
                           name_tag="hqz", nr_layers=2,
                           feature_maps=feature_maps, debug=debug)

        self.qz_t = network(in_dims=intermediate_dims, out_dims=z_net_out_dims,
                            name_tag="qz_t", nr_layers=2,
                            feature_maps=feature_maps * 2, debug=debug)

    @tf.function
    def call(self, x, t, y, step, training=False):
        if self.debug:
            print("Encoding")
        qt_prob = nn.softmax(self.qt_logits(x, step, training=training))
        qt = tfd.Independent(tfd.OneHotCategorical(probs=qt_prob,
                                                   dtype=tf.float64),
                             reinterpreted_batch_ndims=1,
                             name="qt")
        # qt_sample = tf.dtypes.cast(qt.sample(), tf.float64)
        qt_sample = qt.sample()

        hqy = self.hqy(x, step, training=training)
        mu_qy_t = self.mu_qy_t(hqy, step, training=training)

        mu_qy_t = tf.reshape(mu_qy_t, (len(x), *self.y_dims, self.t_dims))
        if training:
            qy_mean = tf.reduce_sum(t * mu_qy_t, axis=-1)
        else:
            qy_mean = tf.reduce_sum(qt_sample * mu_qy_t, axis=-1)

        qy = tfd.Independent(tfd.Normal(loc=qy_mean,
                                        scale=tf.ones_like(qy_mean)),
                             reinterpreted_batch_ndims=1,
                             name="qy")
        if training:
            xy = tf.concat([x, y], -1)
        else:
            xy = tf.concat([x, qy.sample()], -1)

        hidden_z = self.hqz(xy, step, training=training)
        mu_qz_t = self.qz_t(hidden_z, step, training=training)
        mu_qz_t = tf.reshape(mu_qz_t, (len(x), *self.z_dims, self.t_dims))
        qz_mean, qz_std = tf.split(qz0, 2, axis=-2)

        if training:
            qz_mean = tf.reduce_sum(t * qz_mean, axis=-1)
            qz_std = tf.reduce_sum(t * softplus(qz_std), axis=-1)
        else:
            qz_mean = tf.reduce_sum(qt_sample * qz_mean, axis=-1)
            qz_std = tf.reduce_sum(qt_sample * softplus(qz_std), axis=-1)
        return qt_prob, qy_mean, qz_mean, qz_std


class Decoder(Model):

    def __init__(self, x_cat_dims=10, x_cont_dims=10, t_dims=2, y_dims=1,
                 z_dims=32, category_sizes=2, name_tag="no_name",
                 feature_maps=256, architecture_type="FC_net", debug=False):
        """
        Parameters
        ----------

        """

        super().__init__(name=name_tag)
        self.category_sizes = category_sizes

        if architecture_type == "ResNet":
            intermediate_dims = x_cont_dims
            x_cont_out = x_cont_dims[:-1] + (x_cont_dims[-1] * 2, )
            self.x_cat_out_dims = x_cat_dims[:-1] +\
                (x_cat_dims[-1] * category_sizes, )
            self.x_cat_dims = x_cat_dims
        else:
            intermediate_dims = feature_maps
            x_cont_out = x_cont_dims * 2
            self.x_cat_out_dims = x_cat_dims * category_sizes
            self.x_cat_dims = (x_cat_dims, )
        network = eval(architecture_type)

        self.z_dims = z_dims
        self.debug = debug

        self.hx = network(in_dims=z_dims, out_dims=intermediate_dims,
                          name_tag="hx", nr_layers=2,
                          feature_maps=feature_maps, debug=self.debug)

        self.x_cont_logits = network(in_dims=intermediate_dims,
                                     out_dims=x_cont_out, name_tag="x_cont",
                                     nr_layers=2, feature_maps=feature_maps,
                                     debug=self.debug)

        self.x_cat_logits = network(in_dims=intermediate_dims,
                                    out_dims=self.x_cat_out_dims,
                                    name_tag="x_cat", nr_layers=2,
                                    feature_maps=feature_maps,
                                    debug=self.debug)

        self.t_logits = network(in_dims=z_size, out_dims=z_size, name_tag="t",
                                nr_layers=1, feature_maps=feature_maps,
                                squeeze=True, squeeze_dims=t_dims,
                                debug=self.debug)

        self.mu_y_t = network(in_dims=z_size, out_dims=y_dims,
                              name_tag="mu_y_t",
                              nr_layers=2, feature_maps=feature_maps * 2,
                              debug=self.debug)

    @tf.function
    def call(self, z, t, step, training=False):
        if self.debug:
            print("Decoding")
        hidden_x = self.hx(z, step, training=training)
        x_cat_logits = self.x_cat_logits(hidden_x, step, training=training)
        x_cat_prob = nn.softmax(tf.reshape(x_cat_logits,
                                           (len(z), *self.x_cat_dims,
                                            self.category_sizes)))
        x_cat_prob = tf.reshape(x_cat_prob, (len(z), self.x_cat_out_dims,
                                             self.category_sizes))

        x_cont_h = self.x_cont_logits(hidden_x, step, training=training)
        x_cont_mean, x_cont_std = tf.split(x_cont_h, 2, axis=-1)
        x_cont_std = softplus(x_cont_std)

        t_prob = nn.softmax(self.t_logits(z, step, training=training))

        mu_y_t = self.mu_y_t(z, step, training=training)
        y_mean = tf.reduce_sum(t * mu_y_t, axis=-1)

        return x_cat_prob, x_cont_mean, x_cont_std, t_prob, y_mean
        # return x_cont_mean, x_cont_std, t_prob, y_mean

    def do_intervention(self, z, nr_samples):
        """ Computes the quantity E[y|z, do(t=0)] and E[y|z, do(t=1)]

    `   nr_sample of samples are drawn from the Normal distribution
        N(qz_mean, qz_std) and used to infer y for both t=0 and t=1.
        The samples are averaged at the end.

        """
        # y0 = self.mu_y_t0(z, None, training=False)
        # y1 = self.mu_y_t1(z, None, training=False)
        # mu_y0 = tf.reduce_mean(y0, axis=0)
        # mu_y1 = tf.reduce_mean(y1, axis=0)
        # Is this correct? Do we average and sample correctly?

        y = self.mu_y_t(z, None, training=False)
        mean_y = tf.reduce_mean(y, axis=0)

        return mu_y0, mu_y1
