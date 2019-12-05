import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.activations import softplus
from tensorflow_probability import distributions as tfd

from planar_flow import PlanarFlow
from cevae import Encoder, Decoder
from dataset import IHDP_dataset
from utils import get_log_prob, get_analytical_KL_divergence
from evaluation import calc_stats


class CENF(Model):
    
    def __init__(self, params, hidden_size=64, debug=False):
        """ Causal Effect Normalising Flow


        """
        super().__init__()
        self.x_bin_size = params["x_bin_size"]
        self.x_cont_size = params["x_cont_size"]
        self.z_size = params["z_size"]
        self.debug = params["debug"]
        
        self.encode = Encoder(self.x_bin_size, self.x_cont_size, self.z_size, hidden_size, self.debug)
        self.decode = Decoder(self.x_bin_size, self.x_cont_size, self.z_size, hidden_size, self.debug)
        self.planar_flow = PlanarFlow(self.z_size, params["nr_flows"])

    @tf.function
    def call(self, features, step, training=False):
        if self.debug:
            print("Starting forward pass CENF")

        x_bin, x_cont, t, y, y_cf, mu_0, mu_1 = features
        x = tf.concat([x_bin, x_cont], 1)

        encoder_params = self.encode(x, t, y, step, training=training)
        _, _, qz_mean, qz_std = encoder_params
        qz = tfd.Independent(tfd.Normal(loc=qz_mean, scale=qz_std),
                             reinterpreted_batch_ndims=1,
                             name="qz")
        qz_sample = qz.sample()

        qz_k, ldj = self.planar_flow(qz_sample, step, training=training)

        decoder_params = self.decode(qz_k, step, training=training)
        return encoder_params, qz_k, ldj, decoder_params

    @tf.function
    def elbo(self, features, encoder_params, qz_k, ldj, decoder_params, step, params):
        if self.debug:
            print("Calculating loss")
        x_bin, x_cont, t, y, y_cf, mu_0, mu_1 = features
        qt_prob, qy_mean, qz_mean, qz_std = encoder_params
        x_bin_prob, x_cont_mean, x_cont_std, t_prob, y_mean = decoder_params

        ldj = tf.reduce_mean(-ldj)

        if self.debug:
            print("Calculating negative data log likelihood")
        distortion_x = tf.reduce_mean(- get_log_prob(x_bin, 'B', probs=x_bin_prob) \
                           - get_log_prob(x_cont, 'N', mean=x_cont_mean, std=x_cont_std))
        distortion_t = tf.reduce_mean(- get_log_prob(t, 'B', probs=t_prob))
        distortion_y = tf.reduce_mean(- get_log_prob(y, 'N', mean=y_mean))

        if self.debug:
            print("Calculating KL-divergence")
        rate = tf.reduce_mean(get_analytical_KL_divergence(qz_mean, qz_std))

        if self.debug:
            print("Calculating negative log likelihood of auxillary distributions")
        variational_t = - tf.reduce_mean(get_log_prob(t, 'B', probs=qt_prob))
        variational_y = - tf.reduce_mean(get_log_prob(y, 'N', mean=qy_mean))

        if step % params['log_steps'] == 0:
            tf.summary.scalar("distortion/x", distortion_x, step=step)
            tf.summary.scalar("distortion/t", distortion_t, step=step)
            tf.summary.scalar("distortion/y", distortion_y, step=step)
            tf.summary.scalar("rate/z", rate, step=step)
            tf.summary.scalar("rate/ldj", ldj, step=step)
            tf.summary.scalar("variational_ll/t", variational_t, step=step)
            tf.summary.scalar("variational_ll/y", variational_y, step=step)


        elbo_local = -(rate + distortion_x + distortion_t + distortion_y + 
                       variational_t + variational_y + ldj)
        elbo = tf.reduce_mean(input_tensor=elbo_local)
        return -elbo

    def grad(self, features, step, params):
        with tf.GradientTape() as tape:
            encoder_params, qz_k, ldj, decoder_params = self(features, step, training=True)
            loss = self.elbo(features, encoder_params, qz_k, ldj, decoder_params, step, params)
        if self.debug:
            print(f"Forward pass complete, step: {step}")
        return loss, tape.gradient(loss, self.trainable_variables)


