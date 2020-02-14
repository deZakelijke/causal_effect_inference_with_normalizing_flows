import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.activations import softplus
from tensorflow_probability import distributions as tfd

from fc_net import FC_net
from utils import get_log_prob, get_analytical_KL_divergence


class CEVAE(Model):

    def __init__(self, params, hidden_size=200, debug=False):
        """ CEVAE model with fc nets between random variables.
        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py

        After some attempts to port the example code 1 to 1 to TF2, I decided to restructure the
        encoder and decoder to a Model subclass instead to two functions.

        Several fc_nets have an output size of *2 something. This is to output both the mean and 
        std of a Normal distribution at once.
        """
        super().__init__()
        self.x_bin_size = params["x_bin_size"]
        self.x_cont_size = params["x_cont_size"]
        self.z_size = params["z_size"]
        self.debug=params["debug"]
        self.encode = Encoder(self.x_bin_size, self.x_cont_size, self.z_size, hidden_size, self.debug)
        self.decode = Decoder(self.x_bin_size, self.x_cont_size, self.z_size, hidden_size, self.debug)

    @tf.function
    def call(self, features, step, training=False):
        """ Forward pass of the CEVAE

        Args:   features, tuple of all input variables
                step, iteration number of the model for tensorboard logging

        Returns:    encoder_params, parameters of all variational distributions
                    qt_prob, qy_mean, qz_mean, qz_std

                    decoder_params, parameters of all likelihoods
                    x_bin_prob, x_cont_mean, x_cont_std, t_prob, y_mean
        """
        if self.debug:
            print("Starting forward pass")
        x_bin, x_cont, t, y, y_cf, mu_0, mu_1 = features
        x = tf.concat([x_bin, x_cont], 1)

        encoder_params = self.encode(x, t, y, step, training=training)
        _, _, qz_mean, qz_std = encoder_params
        qz = tfd.Independent(tfd.Normal(loc=qz_mean, scale=qz_std),
                             reinterpreted_batch_ndims=1,
                             name="qz")
        qz_sample = qz.sample()

        decoder_params = self.decode(qz_sample, step, training=training)
        return encoder_params, decoder_params

    @tf.function
    def elbo(self, features, encoder_params, decoder_params, step, params):
        if self.debug:
            print("Calculating loss")
        x_bin, x_cont, t, y, y_cf, mu_0, mu_1 = features
        qt_prob, qy_mean, qz_mean, qz_std = encoder_params
        x_bin_prob, x_cont_mean, x_cont_std, t_prob, t_sample, y_mean = decoder_params

        # Get the y values corresponding to the values of t that were sampled during decoding
        t_correct = tf.where(tf.squeeze(t) == tf.squeeze(t_sample))
        t_incorrect = tf.where(tf.squeeze(t) != tf.squeeze(t_sample))
        y_labels = tf.scatter_nd(t_correct, tf.squeeze(tf.gather(y, t_correct), axis=2), y.shape) + \
                   tf.scatter_nd(t_incorrect, tf.squeeze(tf.gather(y_cf, t_incorrect), axis=2), y.shape)

        distortion_x = -get_log_prob(x_bin, 'B', probs=x_bin_prob) \
                       -get_log_prob(x_cont, 'N', mean=x_cont_mean, std=x_cont_std)
        distortion_t = -get_log_prob(t, 'B', probs=t_prob)
        distortion_y = -get_log_prob(y_labels, 'N', mean=y_mean)

        rate = get_analytical_KL_divergence(qz_mean, qz_std)

        variational_t = -get_log_prob(t, 'B', probs=qt_prob)
        variational_y = -get_log_prob(y, 'N', mean=qy_mean)

        if not step is None and step % (params['log_steps']  * 5) == 0:
            l_step = step // (params['log_steps'] * 5)
            tf.summary.scalar("partial_loss/distortion_x", tf.reduce_mean(distortion_x), step=l_step)
            tf.summary.scalar("partial_loss/distortion_t", tf.reduce_mean(distortion_t), step=l_step)
            tf.summary.scalar("partial_loss/distortion_y", tf.reduce_mean(distortion_y), step=l_step)
            tf.summary.scalar("partial_loss/rate_z", tf.reduce_mean(rate), step=l_step)
            tf.summary.scalar("partial_loss/variational_t", tf.reduce_mean(variational_t), step=l_step)
            tf.summary.scalar("partial_loss/variational_y", tf.reduce_mean(variational_y), step=l_step)

        elbo_local = -(params['beta'] * rate + distortion_x + distortion_t + \
                       distortion_y + variational_t + variational_y)
        elbo = tf.reduce_mean(elbo_local)

        return -elbo

    def grad(self, features, step, params):
        with tf.GradientTape() as tape:
            encoder_params, decoder_params = self(features, step, training=True)
            loss = self.elbo(features, encoder_params, decoder_params, step, params)
        if self.debug:
            print(f"Forward pass complete, step: {step}")
        return loss, tape.gradient(loss, self.trainable_variables)

    def do_intervention(self, x, nr_samples):
        _, _, qz_mean, qz_std = self.encode(x, None, None, None, training=False)

        qz = tfd.Independent(tfd.Normal(loc=qz_mean, scale=qz_std),
                             reinterpreted_batch_ndims=1,
                             name="qz")
        z = qz.sample(nr_samples)

        mu_y0, mu_y1 = self.decode.do_intervention(z, nr_samples)
        return mu_y0, mu_y1


class Encoder(Model):

    def __init__(self, x_bin_size, x_cont_size, z_size, hidden_size, debug):
        super().__init__()
        self.x_bin_size = x_bin_size 
        self.x_cont_size = x_cont_size 
        x_size = x_bin_size + x_cont_size
        self.z_size = z_size
        self.debug = debug

        self.qt_logits = FC_net(x_size, 1, "qt", nr_hidden=1, 
                                hidden_size=hidden_size, debug=debug)
        self.hqy       = FC_net(x_size, hidden_size, "hqy",
                                hidden_size=hidden_size, debug=debug)
        self.mu_qy_t0  = FC_net(hidden_size, 1, "mu_qy_t0",
                                hidden_size=hidden_size, debug=debug)
        self.mu_qy_t1  = FC_net(hidden_size, 1, "mu_qy_t1",
                                hidden_size=hidden_size, debug=debug)
        self.hqz       = FC_net(x_size + 1, hidden_size, "hqz", 
                                hidden_size=hidden_size, debug=debug)
        self.qz_t0     = FC_net(hidden_size, z_size * 2, "qz_t0", 
                                hidden_size=hidden_size, debug=debug) 
        self.qz_t1     = FC_net(hidden_size, z_size * 2, "qz_t1", 
                                hidden_size=hidden_size, debug=debug) 

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
        if  training:
            qy_mean = t * mu_qy1 + (1. - t) * mu_qy0
        else:
            qy_mean = qt_sample * mu_qy1 + (1. - qt_sample) * mu_qy0

        qy = tfd.Independent(tfd.Normal(loc=qy_mean, scale=tf.ones_like(qy_mean)),
                             reinterpreted_batch_ndims=1,
                             name="qy")
        
        if training:
            xy = tf.concat([x, y], 1)
        else:
            xy = tf.concat([x, qy.sample()], 1)
        
        hidden_z = self.hqz(xy, step, training=training)
        qz0 = self.qz_t0(hidden_z, step, training=training)
        qz1 = self.qz_t1(hidden_z, step, training=training)

        if training:
            qz_mean = t * qz1[:, :self.z_size] + (1. - t) * qz0[:, :self.z_size]
            qz_std = t * softplus(qz1[:, self.z_size:]) + (1. - t) * softplus(qz0[:, self.z_size:])

        else:
            qz_mean = qt_sample * qz1[:, :self.z_size] + (1. - qt_sample) * qz0[:, :self.z_size]
            qz_std = qt_sample * softplus(qz1[:, self.z_size:]) + (1. - qt_sample) * softplus(qz0[:, self.z_size:])
        return qt_prob, qy_mean, qz_mean, qz_std


class Decoder(Model):

    def __init__(self, x_bin_size, x_cont_size, z_size, hidden_size, debug):
        super().__init__()
        self.x_bin_size = x_bin_size 
        self.x_cont_size = x_cont_size 
        x_size = x_bin_size + x_cont_size
        self.z_size = z_size
        self.debug = debug

        self.hx            = FC_net(z_size, hidden_size, "hx",              
                                    hidden_size=hidden_size, debug=debug)
        self.x_cont_logits = FC_net(hidden_size, x_cont_size * 2, "x_cont", 
                                    hidden_size=hidden_size, debug=debug)
        self.x_bin_logits  = FC_net(hidden_size, x_bin_size, "x_bin",       
                                    hidden_size=hidden_size, debug=debug)
        self.t_logits      = FC_net(z_size, 1, "t", nr_hidden=1, 
                                    hidden_size=hidden_size, debug=debug)
        self.mu_y_t0       = FC_net(z_size, 1, "mu_y_t0", 
                                    hidden_size=hidden_size, debug=debug)
        self.mu_y_t1       = FC_net(z_size, 1, "mu_y_t1", 
                                    hidden_size=hidden_size, debug=debug)

    @tf.function
    def call(self, z, step, training=False):
        if self.debug:
            print("Decoding")
        hidden_x = self.hx(z, step, training=training)
        x_bin_prob = tf.sigmoid(self.x_bin_logits(hidden_x, step, training=training))

        x_cont_h = self.x_cont_logits(hidden_x, step, training=training)
        x_cont_mean = x_cont_h[:, :self.x_cont_size]
        x_cont_std = softplus(x_cont_h[:, self.x_cont_size:])

        t_prob = tf.sigmoid(self.t_logits(z, step, training=training))
        t = tfd.Independent(tfd.Bernoulli(probs=t_prob),
                            reinterpreted_batch_ndims=1,
                            name="t")

        t_sample = tf.dtypes.cast(t.sample(), tf.float64)

        mu_y0 = self.mu_y_t0(z, step, training=training)
        mu_y1 = self.mu_y_t1(z, step, training=training)
        y_mean = t_sample * mu_y1 + (1. - t_sample) * mu_y0
        return x_bin_prob, x_cont_mean, x_cont_std, t_prob, t_sample, y_mean

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

