import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.activations import softplus
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow_probability import distributions as tfd

from cevae import Encoder, Decoder
from evaluation import calc_stats
from planar_flow import PlanarFlow
from utils import get_log_prob, get_analytical_KL_divergence


class CENF(Model):
    """ Causal Effect Normalising Flow

    """

    def __init__(
        self,
        x_cat_dims=10,
        x_cont_dims=10,
        t_dims=2,
        y_dims=1,
        z_dims=32,
        category_sizes=2,
        t_loss=CategoricalCrossentropy(),
        y_loss=MeanSquaredError(),
        n_flows=2,
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
        super().__init__()
        self.debug = debug
        self.category_sizes = category_sizes
        self.t_dims = t_dims
        self.t_loss = t_loss
        self.y_dims = y_dims
        self.y_loss = y_loss
        self.log_steps = log_steps
        self.architecture_type = architecture_type

        self.annealing_factor = 1e-8

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
        self.z_flow = PlanarFlow(z_dims, n_flows)

    @tf.function
    def call(self, x, t, y, step, training=False):
        if self.debug:
            print("Starting forward pass CENF")

        encoder_params = self.encode(x, t, y, step, training=training)
        _, _, qz_mean, qz_std = encoder_params
        qz = tf.random.normal(qz_mean.shape, dtype=tf.float64)
        qz = qz * qz_std + qz_mean

        qz_k, ldj = self.z_flow(qz, step, training=training)

        decoder_params = self.decode(qz_k, t, step, training=training)
        return encoder_params, qz_k, ldj, decoder_params

    @tf.function
    def loss(self, features, encoder_params, qz_k, ldj_z, decoder_params,
             step):
        if self.debug:
            print("Calculating loss")
        x_cat, x_cont, t, _, y, *_ = features
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
            tf.summary.scalar("partial_loss/ldj",
                              tf.reduce_mean(-ldj_z), step=l_step)
            tf.summary.scalar("partial_loss/variational_t",
                              tf.reduce_mean(variational_t), step=l_step)
            tf.summary.scalar("partial_loss/variational_y",
                              tf.reduce_mean(variational_y), step=l_step)

        elbo_local = -(self.annealing_factor * rate + distortion_x +
                       distortion_t + distortion_y +
                       variational_t + variational_y - ldj_z)
        elbo = tf.reduce_mean(input_tensor=elbo_local)
        return -elbo

    def do_intervention(self, x, nr_samples):
        *_, qz_mean, qz_std = self.encode(x, None, None, None, training=False)
        final_shape = (nr_samples, qz_mean.shape[0], self.y_dims, self.t_dims)
        qz = tf.random.normal((nr_samples, *qz_mean.shape), dtype=tf.float64)
        z = qz * qz_std + qz_mean
        z_k, ldj = self.z_flow(z, None, training=False)
        y = self.decode.do_intervention(z_k, nr_samples)
        y_mean = tf.reduce_mean(tf.reshape(y, final_shape), axis=0)
        mu_y0, mu_y1 = y_mean[..., 0], y_mean[..., 1]
        return mu_y0, mu_y1


if __name__ == "__main__":
    pass
