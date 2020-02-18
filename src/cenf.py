import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.activations import softplus
from tensorflow_probability import distributions as tfd

from planar_flow import PlanarFlow
from fc_net import FC_net
from cevae import Encoder, Decoder
from dataset import IHDP_dataset
from utils import get_log_prob, get_analytical_KL_divergence
from evaluation import calc_stats


class CENF(Model):

    def __init__(self, params, hidden_size=200, debug=False):
        """ Causal Effect Normalising Flow


        """
        super().__init__()
        self.x_bin_size = params["x_bin_size"]
        self.x_cont_size = params["x_cont_size"]
        self.z_size = params["z_size"]
        self.debug = params["debug"]
        self.flow_decoder = True
        
        self.encode = Encoder(self.x_bin_size, self.x_cont_size, self.z_size, hidden_size, self.debug)
        if self.flow_decoder:
            self.decode = FlowDecoder(self.x_bin_size, self.x_cont_size, 
                                      self.z_size, hidden_size, params['nr_flows'], self.debug)
        else:
            self.decode = Decoder(self.x_bin_size, self.x_cont_size, 
                                  self.z_size, hidden_size, self.debug)
        self.z_flow = PlanarFlow(self.z_size, params["nr_flows"])

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

        qz_k, ldj = self.z_flow(qz_sample, step, training=training)

        decoder_params = self.decode(qz_k, step, training=training)
        return encoder_params, qz_k, ldj, decoder_params

    @tf.function
    def elbo(self, features, encoder_params, qz_k, ldj_z, decoder_params, step, params):
        if self.debug:
            print("Calculating loss")
        x_bin, x_cont, t, y, y_cf, mu_0, mu_1 = features
        qt_prob, qy_mean, qz_mean, qz_std = encoder_params

        if self.flow_decoder:
            x_bin_prob, x_cont_mean, x_cont_std, t_prob, t_sample, y_mean, yK, ldj_y = decoder_params
        else:
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
        ldj_z = tf.reduce_mean(-ldj_z)
        if self.flow_decoder:
            ldj_y = tf.reduce_mean(-ldj_y)

        variational_t = -get_log_prob(t, 'B', probs=qt_prob)
        variational_y = -get_log_prob(y, 'N', mean=qy_mean)

        if not step is None and step % (params['log_steps'] * 5) == 0:
            l_step = step // (params['log_steps'] * 5)
            tf.summary.scalar("partial_loss/distortion_x", tf.reduce_mean(distortion_x), step=l_step)
            tf.summary.scalar("partial_loss/distortion_t", tf.reduce_mean(distortion_t), step=l_step)
            tf.summary.scalar("partial_loss/distortion_y", tf.reduce_mean(distortion_y), step=l_step)
            tf.summary.scalar("partial_loss/rate_z", tf.reduce_mean(rate), step=l_step)
            tf.summary.scalar("partial_loss/ldj_z", ldj_z, step=l_step)
            if self.flow_decoder:
                tf.summary.scalar("partial_loss/ldj_y", ldj_y, step=l_step)
            tf.summary.scalar("partial_loss/variational_t", tf.reduce_mean(variational_t), step=l_step)
            tf.summary.scalar("partial_loss/variational_y", tf.reduce_mean(variational_y), step=l_step)


        if self.flow_decoder:
            elbo_local = -(rate + distortion_x + distortion_t + distortion_y + \
                           variational_t + variational_y + ldj_z + ldj_y)
        else:
            elbo_local = -(rate + distortion_x + distortion_t + distortion_y + \
                           variational_t + variational_y + ldj_z)
        elbo = tf.reduce_mean(input_tensor=elbo_local)
        return -elbo

    def grad(self, features, step, params):
        with tf.GradientTape() as tape:
            encoder_params, qz_k, ldj, decoder_params = self(features, step, training=True)
            loss = self.elbo(features, encoder_params, qz_k, ldj, decoder_params, step, params)
        if self.debug:
            print(f"Forward pass complete, step: {step}")
        return loss, tape.gradient(loss, self.trainable_variables)

    def do_intervention(self, x, nr_samples):
        _, _, qz_mean, qz_std = self.encode(x, None, None, None, training=False)
        qz = tfd.Independent(tfd.Normal(loc=qz_mean, scale=qz_std),
                             reinterpreted_batch_ndims=1,
                             name="qz")
        qz_sample = qz.sample(nr_samples)

        qz_k, ldj = self.z_flow(qz_sample, None, training=False)

        mu_y0, mu_y1 = self.decode.do_intervention(qz_k, nr_samples)
        return mu_y0, mu_y1

class FlowDecoder(Model):
    """ A variation of the decoder that models p(y|t,z) with a normalising flow.

    This class is very similar to the Decoder of the cevae and the first version of
    the cenf. The main difference is that we no longer use two MLPs to model
    p(y|t=1,z) and p(y|t=0,z), but use one MLP for both and then use a Normalising Flow
    to capture a more complex distribution than a diagonal Gaussian with unit covariance.

    """

    def __init__(self, x_bin_size, x_cont_size, z_size, hidden_size, nr_flows, debug):
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
        self.mu_y0         = FC_net(z_size + 1, 1, "mu_y0",
                                    hidden_size=hidden_size, debug=debug)
        self.y_flow        = PlanarFlow(1, nr_flows)

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

        y0 = self.mu_y0(tf.concat([z, t_sample], 1), step, training=training)
        yK, ldj = self.y_flow(y0, step, training=training)
        return x_bin_prob, x_cont_mean, x_cont_std, t_prob, t_sample, y0, yK, ldj



    def do_intervention(self, z, nr_samples):
        t0 = tf.zeros((z.shape[0], z.shape[1], 1), dtype=tf.float64)
        t1 = tf.ones((z.shape[0], z.shape[1], 1), dtype=tf.float64)

        y0_t0 = self.mu_y0(tf.concat([z, t0], 2), None, training=False)
        y0_t1 = self.mu_y0(tf.concat([z, t1], 2), None, training=False)

        yK_t0, ldj_t0 = self.y_flow(y0_t0, None, training=False)
        yK_t1, ldj_t1 = self.y_flow(y0_t1, None, training=False)

        mu_y_t0 = tf.reduce_mean(yK_t0, axis=0)
        mu_y_t1 = tf.reduce_mean(yK_t1, axis=0)
        return mu_y_t0, mu_y_t1


if __name__ == "__main__":
    pass
