import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.activations import softplus
from tensorflow_probability import distributions as tfd

from cevae import Encoder, Decoder
from dataset import IHDP_dataset
from evaluation import calc_stats
from fc_net import FC_net
from planar_flow import PlanarFlow
from utils import get_log_prob, get_analytical_KL_divergence


class CENF(Model):

    def __init__(self, params, category_sizes, hidden_size=200, debug=False):
        """ Causal Effect Normalising Flow


        """
        super().__init__()
        self.debug = debug
        self.category_sizes = category_sizes

        self.encode = Encoder(params, category_sizes, hidden_size)
        self.decode = Decoder(params, category_sizes, hidden_size)
        self.z_flow = PlanarFlow(params["z_size"], params["n_flows"])

    @tf.function
    def call(self, features, step, training=False):
        if self.debug:
            print("Starting forward pass CENF")

        x_cat, x_cont, t, y, y_cf, mu_0, mu_1 = features
        x = tf.concat([x_cat, x_cont], 1)

        encoder_params = self.encode(x, t, y, step, training=training)
        _, _, qz_mean, qz_std = encoder_params
        qz = tfd.Independent(tfd.Normal(loc=qz_mean, scale=qz_std),
                             reinterpreted_batch_ndims=1,
                             name="qz")
        qz_sample = qz.sample()

        qz_k, ldj = self.z_flow(qz_sample, step, training=training)

        decoder_params = self.decode(qz_k, t, step, training=training)
        return encoder_params, qz_k, ldj, decoder_params

    @tf.function
    def loss(self, features, encoder_params, qz_k, ldj_z, decoder_params,
             step, params):
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

        distortion_x = -get_log_prob(x_cat, 'M', probs=x_cat_prob) \
                       - get_log_prob(x_cont, 'N', mean=x_cont_mean,
                                      std=x_cont_std)
        distortion_t = -get_log_prob(t, 'B', probs=t_prob)
        distortion_y = -get_log_prob(y, 'N', mean=y_mean)

        rate = get_analytical_KL_divergence(qz_mean, qz_std)
        # ldj_z = tf.reduce_mean(-ldj_z)

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
            tf.summary.scalar("partial_loss/ldj_z",
                              tf.reduce_mean(-ldj_z), step=l_step)
            tf.summary.scalar("partial_loss/variational_t",
                              tf.reduce_mean(variational_t), step=l_step)
            tf.summary.scalar("partial_loss/variational_y",
                              tf.reduce_mean(variational_y), step=l_step)

        elbo_local = -(rate + distortion_x + distortion_t + distortion_y +
                       variational_t + variational_y - ldj_z)
        elbo = tf.reduce_mean(input_tensor=elbo_local)
        return -elbo

    def do_intervention(self, x, nr_samples):
        *_, qz_mean, qz_std = self.encode(x, None, None, None, training=False)
        qz = tfd.Independent(tfd.Normal(loc=qz_mean, scale=qz_std),
                             reinterpreted_batch_ndims=1,
                             name="qz")
        qz_sample = qz.sample(nr_samples)

        qz_k, ldj = self.z_flow(qz_sample, None, training=False)

        mu_y0, mu_y1 = self.decode.do_intervention(qz_k, nr_samples)
        return mu_y0, mu_y1

if __name__ == "__main__":
    pass
