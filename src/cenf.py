import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.activations import softplus
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow_probability import distributions as tfd

from cevae import CEVAE, Encoder, Decoder
from evaluation import calc_stats
from utils import get_log_prob, get_analytical_KL_divergence


class CENF(CEVAE):
    """ Causal Effect Normalising Flow

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
        n_flows=2,
        name_tag="no_name",
        feature_maps=256,
        architecture_type="FC_net",
        log_steps=10,
        flow_type_variational="PlanarFlow",
        debug=False,
        **_
    ):
        """
        Parameters
        ----------

        """
        super().__init__(
            x_dims=x_dims,
            x_cat_dims=x_cat_dims,
            x_cont_dims=x_cont_dims,
            t_dims=t_dims,
            t_type=t_type,
            y_dims=y_dims,
            y_type=y_type,
            z_dims=z_dims,
            category_sizes=category_sizes,
            name_tag=name_tag,
            feature_maps=feature_maps,
            architecture_type=architecture_type,
            log_steps=log_steps,
            debug=debug
        )
        self.log_steps = log_steps

    @tf.function
    def call(self, x, t, y, step, training=False):
        if self.debug:
            print(f"Starting forward pass {self.name_tag}")

        encoder_params = self.encoder(x, t, y, step, training=training)
        _, _, qz_mean, qz_std = encoder_params
        qz = tf.random.normal(qz_mean.shape, dtype=tf.float64)
        qz = qz * qz_std + qz_mean

        # qz_k, ldj = self.z_flow(qz, step, training=training)
        z_shape = qz.shape
        ldj = 0.0
        qz = self.flow_start_layer(qz)
        for i in range(self.n_flows):
            ldj += self.z_flows[i].logdet_jacobian(qz)
            qz =self.z_flows[i](qz, step, training=training)
        qz_k = tf.reshape(qz, z_shape)

        decoder_params = self.decoder(qz_k, t, step, training=training)
        return encoder_params, qz_k, ldj, decoder_params

    @tf.function
    def loss(self, features, encoder_params, qz_k, ldj_z, decoder_params,
             step):
        if self.debug:
            print("Calculating loss")
        encoder_loss = self.encoder.loss(features, encoder_params, step)
        decoder_loss = self.decoder.loss(features, decoder_params, step)

        if step is not None and step % (self.log_steps * 10) == 0:
            l_step = step // (self.log_steps * 10)
            tf.summary.scalar("partial_loss/ldj",
                              tf.reduce_mean(-ldj_z), step=l_step)

        elbo_local =  encoder_loss + decoder_loss + ldj_z
        elbo = tf.reduce_mean(input_tensor=elbo_local)
        return -elbo

    def do_intervention(self, x, t0, t1, nr_samples):
        *_, qz_mean, qz_std = self.encoder(x, None, None, None, training=False)
        qz = tf.random.normal((nr_samples, *qz_mean.shape), dtype=tf.float64)
        z = qz * qz_std + qz_mean

        z_shape = z.shape
        ldj = 0.0
        for i in range(self.n_flows):
            ldj += self.z_flows[i].logdet_jacobian(z)
            z =self.z_flows[i](z, None, training=False)
        z_k = tf.reshape(z, z_shape)

        y0, y1 = self.decoder.do_intervention(z_k, t0, t1, nr_samples)
        return y0, y1


if __name__ == "__main__":
    pass
