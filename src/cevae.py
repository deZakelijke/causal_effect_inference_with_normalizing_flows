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

# TODO al die shapes bullshit
# Add extra output to make t one-hot encoded

class CEVAE(Model):

    def __init__(self, params, category_sizes, hidden_size=200, debug=False):
        """ CEVAE model with fc nets between random variables.
        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py

        After some attempts to port the example code 1 to 1 to TF2, I decided
        to restructure the encoder and decoder to a Model subclass instead to
        two functions.

        Several fc_nets have an output size of *2 something. This is to output
        both the mean and std of a Normal distribution at once.
        """
        super().__init__()
        self.debug = debug
        self.category_sizes = category_sizes
        self.encode = Encoder(params, category_sizes, hidden_size)
        self.decode = Decoder(params, category_sizes, hidden_size)

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
        x = tf.concat([x_cat, x_cont], 1)

        encoder_params = self.encode(x, t, y, step, training=training)
        _, _, qz_mean, qz_std = encoder_params
        qz = tf.random.normal(qz_mean.shape, dtype=tf.float64)
        qz = qz * qz_std + qz_mean

        decoder_params = self.decode(qz, t, step, training=training)
        return encoder_params, decoder_params

    @tf.function
    def loss(self, features, encoder_params, decoder_params, step, params):
        if self.debug:
            print("Calculating loss")
        x_cat, x_cont, t, y, y_cf, mu_0, mu_1 = features
        qt_prob, qy_mean, qz_mean, qz_std = encoder_params
        x_cat_prob, x_cont_mean, x_cont_std, t_prob, y_mean = decoder_params
        l, f = x_cat.shape
        x_cat_prob = tf.reshape(x_cat_prob, (l, f//self.category_sizes,
                                             self.category_sizes))
        x_cat = tf.reshape(x_cat, (l, f//self.category_sizes,
                                   self.category_sizes))

        y_type = params['dataset_distributions']['y']
        distortion_x = -get_log_prob(x_cat, 'M', probs=x_cat_prob) \
                       - get_log_prob(x_cont, 'N', mean=x_cont_mean,
                                      std=x_cont_std)
        distortion_t = -get_log_prob(t, 'B', probs=t_prob)
        distortion_y = -get_log_prob(y, y_type, mean=y_mean, probs=y_mean)

        rate = get_analytical_KL_divergence(qz_mean, qz_std)

        variational_t = -get_log_prob(t, 'B', probs=qt_prob)
        variational_y = -get_log_prob(y, 'N', mean=qy_mean)

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

        elbo_local = -(rate + distortion_x + distortion_t + distortion_y +
                       variational_t + variational_y)
        elbo = tf.reduce_mean(elbo_local)
        return -elbo

    def do_intervention(self, x, nr_samples):
        _, _, qz_mean, qz_std = self.encode(x, None, None, None,
                                            training=False)

        qz = tf.random.normal((nr_samples, *qz_mean.shape), dtype=tf.float64)
        z = qz * qz_std + qz_mean

        # qz = tfd.Independent(tfd.Normal(loc=qz_mean, scale=qz_std),
        #                      reinterpreted_batch_ndims=1,
        #                      name="qz")
        # z = qz.sample(nr_samples)

        mu_y0, mu_y1 = self.decode.do_intervention(z, nr_samples)
        return mu_y0, mu_y1


class Encoder(Model):

    def __init__(self, params, category_sizes, hidden_size):
        super().__init__()
        self.x_cat_size = params["x_cat_size"]
        x_cont_size = params["x_cont_size"]
        z_size = params["z_size"]
        y_size = params["y_size"]
        self.debug = params["debug"]

        if params['dataset'] == "SHAPES":
            x_size = x_cont_size
            intermediate_size = x_size
            z_in_size = x_size[:-1] + (x_size[-1] + y_size[-1], )
            z_out_size = z_size[:-1] + (z_size[-1] * 2, )
            architecture_type = "ResNet"
        else:
            x_size = self.x_cat_size * category_sizes + x_cont_size
            intermediate_size = hidden_size
            z_in_size = x_size + y_size
            z_out_size = z_size * 2
            architecture_type = "FC_net"
        network = eval(architecture_type)

        self.qt_logits = network(x_size, x_size, "qt", 1,
                                 hidden_size, squeeze=True, squeeze_dims=1,
                                 debug=self.debug)
        self.hqy = network(x_size, intermediate_size, "hqy", 2,
                           hidden_size, debug=self.debug)

        self.mu_qy_t0 = network(intermediate_size, y_size, "mu_qy_t0", 2,
                                hidden_size, debug=self.debug)
        self.mu_qy_t1 = network(intermediate_size, y_size, "mu_qy_t1", 2,
                                hidden_size, debug=self.debug)

        self.hqz = network(z_in_size, intermediate_size, "hqz", 2,
                           hidden_size, debug=self.debug)
        self.qz_t0 = network(intermediate_size, z_out_size, "qz_t0", 2,
                             hidden_size, debug=self.debug)
        self.qz_t1 = network(intermediate_size, z_out_size, "qz_t1", 2,
                             hidden_size, debug=self.debug)

    @tf.function
    def call(self, x, t, y, step, training=False):
        if self.debug:
            print("Encoding")
        qt_prob = tf.sigmoid(self.qt_logits(x, step, training=training))
        qt = tfd.Independent(tfd.Bernoulli(probs=qt_prob),
                             reinterpreted_batch_ndims=1,
                             name="qt")
        qt_sample = tf.dtypes.cast(qt.sample(), tf.float64)

        hqy = self.hqy(x, step, training=training)
        mu_qy0 = self.mu_qy_t0(hqy, step, training=training)
        mu_qy1 = self.mu_qy_t1(hqy, step, training=training)
        if training:
            qy_mean = t * mu_qy1 + (1. - t) * mu_qy0
        else:
            qy_mean = qt_sample * mu_qy1 + (1. - qt_sample) * mu_qy0

        qy = tfd.Independent(tfd.Normal(loc=qy_mean,
                                        scale=tf.ones_like(qy_mean)),
                             reinterpreted_batch_ndims=1,
                             name="qy")

        if training:
            xy = tf.concat([x, y], -1)
        else:
            xy = tf.concat([x, qy.sample()], -1)

        hidden_z = self.hqz(xy, step, training=training)
        qz0 = self.qz_t0(hidden_z, step, training=training)
        qz1 = self.qz_t1(hidden_z, step, training=training)
        qz0_mean, qz0_std = tf.split(qz0, 2, axis=-1)
        qz1_mean, qz1_std = tf.split(qz1, 2, axis=-1)

        if training:
            qz_mean = t * qz1_mean + (1. - t) *\
                      qz0_mean
            qz_std = t * softplus(qz1_std) + (1. - t) *\
                softplus(qz0_std)

        else:
            qz_mean = qt_sample * qz1_mean + (1. - qt_sample) *\
                      qz0_mean
            qz_std = qt_sample * softplus(qz1_std) +\
                (1. - qt_sample) * softplus(qz0_std)
        return qt_prob, qy_mean, qz_mean, qz_std


class Decoder(Model):

    def __init__(self, params, category_sizes, hidden_size):
        super().__init__()
        self.x_cat_size = params["x_cat_size"]
        self.category_sizes = category_sizes
        x_cat_shape = self.x_cat_size * category_sizes
        x_cont_size = params["x_cont_size"]

        if params['dataset'] == "SHAPES":
            intermediate_size = x_cont_size
            x_cont_in = x_cont_size[:-1] + (x_cont_size[-1] * 2, )
            architecture_type = "ResNet"
        else:
            intermediate_size = hidden_size
            x_cont_in = x_cont_size * 2
            architecture_type = "FC_net"
            self.x_cat_size = (self.x_cat_size, )
        network = eval(architecture_type)

        self.z_size = params["z_size"]
        self.debug = params["debug"]
        y_size = params["y_size"]

        self.hx = network(self.z_size, intermediate_size, "hx", 2,
                          hidden_size, debug=self.debug)
        self.x_cont_logits = network(intermediate_size,
                                     x_cont_in, "x_cont", 2,
                                     hidden_size, debug=self.debug)
        self.x_cat_logits = network(intermediate_size, x_cat_shape, "x_cat",
                                    2, hidden_size, debug=self.debug)
        self.t_logits = network(self.z_size, self.z_size, "t", 1,
                                hidden_size, squeeze=True,
                                squeeze_dims=1, debug=self.debug)
        self.mu_y_t0 = network(self.z_size, y_size, "mu_y_t0", 2,
                               hidden_size, debug=self.debug)
        self.mu_y_t1 = network(self.z_size, y_size, "mu_y_t1", 2,
                               hidden_size, debug=self.debug)

    @tf.function
    def call(self, z, t, step, training=False):
        if self.debug:
            print("Decoding")
        hidden_x = self.hx(z, step, training=training)
        x_cat_logits = self.x_cat_logits(hidden_x, step, training=training)
        x_cat_prob = nn.softmax(tf.reshape(x_cat_logits,
                                           (len(x_cat_logits), *self.x_cat_size,
                                            self.category_sizes)))
        x_cat_prob = tf.reshape(x_cat_prob, (len(x_cat_prob),
                                             self.x_cat_size[-1] *
                                             self.category_sizes))

        x_cont_h = self.x_cont_logits(hidden_x, step, training=training)
        x_cont_mean, x_cont_std = tf.split(x_cont_h, 2, axis=-1)

        t_prob = tf.sigmoid(self.t_logits(z, step, training=training))

        mu_y0 = self.mu_y_t0(z, step, training=training)
        mu_y1 = self.mu_y_t1(z, step, training=training)
        y_mean = t * mu_y1 + (1. - t) * mu_y0
        return x_cat_prob, x_cont_mean, x_cont_std, t_prob, y_mean

    def do_intervention(self, z, nr_samples):
        """ Computes the quantity E[y|z, do(t=0)] and E[y|z, do(t=1)]

    `   nr_sample of samples are drawn from the Normal distribution
        N(qz_mean, qz_std) and used to infer y for both t=0 and t=1.
        The samples are averaged at the end.

        """
        y0 = self.mu_y_t0(z, None, training=False)
        y1 = self.mu_y_t1(z, None, training=False)
        mu_y0 = tf.reduce_mean(y0, axis=0)
        mu_y1 = tf.reduce_mean(y1, axis=0)
        # Is this correct? Do we average and sample correctly?

        return mu_y0, mu_y1
