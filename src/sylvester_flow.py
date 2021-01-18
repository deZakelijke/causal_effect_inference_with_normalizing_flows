import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.activations import softplus
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow import math
from tensorflow import nn
from tensorflow_probability import distributions as tfd

from cenf import CENF
from cevae import Encoder
from utils import get_analytical_KL_divergence


class SylvesterFlow(CENF):
    """ Sylvester flow model

    Implementation of Sylvester Normalising Flow, as described in
    https://arxiv.org/abs/1803.05649 by van den Berg et al.
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
        householder_maps=8,
        name_tag="Sylvester flow",
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
        z_dims : int
            Number of latent dimensions of the flow.
        n_flows : int
            Number of planar flows in the model.
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

        assert n_flows >= 0 and type(n_flows) == int,\
            "Number of flows must be larger than 0"
        self.n_flows = n_flows
        self.householder_maps = householder_maps
        if type(z_dims) == tuple:
            self.first_layer = layers.Flatten()
        else:
            self.first_layer = tf.identity
        z_dims = tf.cast(tf.reduce_prod(z_dims), tf.int32).numpy()
        self.z_dims = z_dims

        if t_type == "Categorical":
            t_loss = CategoricalCrossentropy()
            t_dist = lambda x: tfd.OneHotCategorical(probs=x,
                                                     dtype=tf.float64)
            t_activation = nn.softmax
        else:
            t_loss = MeanSquaredError()
            t_dist = lambda x: tfd.Normal(x, scale=tf.ones_like(x))
            t_activation = lambda x: x

        if y_type == "Categorical":
            y_loss = CategoricalCrossentropy()
            y_dist = lambda x: tfd.OneHotCategorical(probs=x,
                                                     dtype=tf.float64)
            y_activation = nn.softmax
        else:
            y_loss = MeanSquaredError()
            y_dist = lambda x: tfd.Normal(x, scale=tf.ones_like(x))
            y_activation = lambda x: x

        self.encoder = SylvesterEncoder(x_dims, t_dims, t_dist, t_loss,
                                        t_activation, y_dims, y_dist, y_loss,
                                        y_activation, z_dims, "Encoder",
                                        feature_maps, architecture_type,
                                        n_flows, householder_maps, debug)

    @tf.function
    def call(self, x, t, y, step, training=False):
        if self.debug:
            print(f"Starting forward pass Sylvester flow")

        encoder_params = self.encoder(x, t, y, step, training=training)
        _, _, qz_mean, qz_std, R1, R2, Q, b = encoder_params
        qz = tf.random.normal(qz_mean.shape, dtype=tf.float64)
        qz = qz * qz_std + qz_mean
        z_shape = qz.shape
        ldj = 0.0

        for i in range(self.n_flows):
            qz, ldj_i = self.single_flow_pass(qz, R1[..., i], R2[..., i],
                                              Q[:, i], b[..., i])
            ldj += ldj_i
        qz_k = tf.reshape(qz, z_shape)

        decoder_params = self.decoder(qz_k, t, step, training=training)
        return encoder_params, qz_k, ldj * 0.001, decoder_params

    @staticmethod
    def tanh_deriv(x):
        return 1 - tf.tanh(x) ** 2

    def single_flow_pass(self, z, R1, R2, Q, b):
        diag_r1 = tf.linalg.diag_part(R1)
        diag_r2 = tf.linalg.diag_part(R2)

        qr2 = Q @ tf.transpose(R2, [0, 2, 1])
        qr1 = Q @ R1
        r2qzb = tf.einsum('ij,ijk->ik', z, qr2) + b
        z = tf.einsum('ij,ijk->ik', tf.tanh(r2qzb),
                      tf.transpose(qr1, [0, 2, 1]))

        diag_j = diag_r1 * diag_r2
        diag_j = self.tanh_deriv(r2qzb) * diag_j + 1.
        ldj = tf.reduce_sum(math.log(math.abs(diag_j)), axis=1)

        return z, ldj

    def do_intervention(self, x, t0, t1, nr_samples):
        encoder_params = self.encoder(x, None, None, None, training=False)
        _, _, qz_mean, qz_std, R1, R2, Q, b = encoder_params
        qz = tf.random.normal((nr_samples, *qz_mean.shape), dtype=tf.float64)
        qz = qz * qz_std + qz_mean
        z_shape = qz.shape

        qz = tf.reshape(qz, [-1, self.z_dims])
        R1_stack = tf.tile(R1, [nr_samples, 1, 1, 1])
        R2_stack = tf.tile(R2, [nr_samples, 1, 1, 1])
        Q_stack = tf.tile(Q, [nr_samples, 1, 1, 1])
        b_stack = tf.tile(b, [nr_samples, 1, 1])

        for i in range(self.n_flows):
            qz, _ = self.single_flow_pass(qz, R1_stack[..., i],
                                          R2_stack[..., i], Q_stack[:, i],
                                          b_stack[..., i])
        qz_k = tf.reshape(qz, z_shape)
        y0, y1 = self.decoder.do_intervention(qz_k, t0, t1, nr_samples)
        return y0, y1


class SylvesterEncoder(Encoder):
    """ Special encoder for the Sylvester flow.

    This encoder also outputs the Q and R values for each layer of the
    sylvester flow.
    """
    def __init__(
        self,
        x_dims=30,
        t_dims=2,
        t_dist=None,
        t_loss=None,
        t_activation=None,
        y_dims=1,
        y_dist=None,
        y_loss=None,
        y_activation=None,
        z_dims=32,
        name_tag="no_name",
        feature_maps=200,
        architecture_type="FC_net",
        n_flows=4,
        householder_maps=8,
        debug=False
    ):

        super().__init__(
            x_dims=x_dims,
            t_dims=t_dims,
            t_dist=t_dist,
            t_loss=t_loss,
            t_activation=t_activation,
            y_dims=y_dims,
            y_dist=y_dist,
            y_loss=y_loss,
            y_activation=y_activation,
            z_dims=z_dims,
            name_tag=name_tag,
            feature_maps=feature_maps,
            architecture_type=architecture_type,
            debug=debug
        )

        self.n_flows = n_flows
        self.householder_maps = householder_maps
        self.diag_idx = tf.range(z_dims)
        self.eye = tf.eye(z_dims, batch_shape=[1])

        triu_mask = tf.ones((z_dims, z_dims), dtype=tf.float64)
        triu_mask = triu_mask - tf.linalg.band_part(triu_mask, -1, 0)
        self.triu_mask = tf.reshape(triu_mask, [1, z_dims, z_dims, 1])

        self.Q_map = layers.Dense(z_dims * n_flows * householder_maps)
        self.b_map = layers.Dense(z_dims * n_flows)
        self.tri_map = layers.Dense(z_dims * z_dims * n_flows)
        self.diag_map1 = layers.Dense(z_dims * n_flows, activation='tanh')
        self.diag_map2 = layers.Dense(z_dims * n_flows, activation='tanh')

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

        if training:
            qy_mean = self.mu_qy_t(tf.concat([hqy, t], -1), step,
                                   training=training)
        else:
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
        mu_qz_t = self.qz_t(xyt, step, training=training)
        qz_mean, qz_std = tf.split(mu_qz_t, 2, axis=-1)
        qz_std = softplus(qz_std)

        triangular = self.tri_map(xyt)
        diag1 = self.diag_map1(xyt)
        diag2 = self.diag_map2(xyt)

        triangular = tf.reshape(triangular,
                                [-1, self.z_dims, self.z_dims, self.n_flows])
        diag1 = tf.reshape(diag1, [-1, self.z_dims, self.n_flows])
        diag2 = tf.reshape(diag2, [-1, self.z_dims, self.n_flows])

        R1 = triangular * self.triu_mask
        R2 = tf.transpose(triangular, perm=[0, 2, 1, 3]) * self.triu_mask

        R1 = tf.transpose(tf.linalg.set_diag(tf.transpose(R1, [0, 3, 1, 2]),
                                             tf.transpose(diag1, [0, 2, 1])),
                          [0, 2, 3, 1])
        R2 = tf.transpose(tf.linalg.set_diag(tf.transpose(R2, [0, 3, 1, 2]),
                                             tf.transpose(diag2, [0, 2, 1])),
                          [0, 2, 3, 1])

        Q = self.Q_map(xyt)
        Q = self.batch_construct_orthogonal(Q)

        b = self.b_map(xyt)
        b = tf.reshape(b, [-1, self.z_dims, self.n_flows])
        return qt_prob, qy_mean, qz_mean, qz_std, R1, R2, Q, b

    def batch_construct_orthogonal(self, Q):
        """ Batch construction of orthogonal matrix"""
        Q = tf.reshape(Q, [-1, self.z_dims])
        norm = tf.norm(Q, axis=1, keepdims=True)
        v = Q / norm
        vvT = tf.einsum('ij,ik->ijk', v, v)

        amat = self.eye = vvT
        amat = tf.reshape(amat, [-1, self.householder_maps,
                                 self.z_dims, self.z_dims])

        tmp = amat[:, 0]
        for k in tf.range(1, self.householder_maps):
            tmp = amat[:, k] @ tmp
        Q = tf.reshape(tmp, [-1, self.n_flows, self.z_dims, self.z_dims])
        return Q

    @tf.function
    def loss(self, features, encoder_params, step):
        x_cat, x_cont, t, _, y, *_ = features
        qt_prob, qy_mean, qz_mean, qz_std, _, _, _, _, = encoder_params
        rate = get_analytical_KL_divergence(qz_mean, qz_std)
        variational_t = self.t_loss(t, qt_prob)
        variational_y = self.y_loss(y, qy_mean)
        if step is not None and step % (self.log_steps * 10) == 0:
            l_step = step // (self.log_steps * 10)
            tf.summary.scalar("partial_loss/rate_z",
                              tf.reduce_mean(rate), step=l_step)
            tf.summary.scalar("partial_loss/variational_t",
                              tf.reduce_mean(variational_t), step=l_step)
            tf.summary.scalar("partial_loss/variational_y",
                              tf.reduce_mean(variational_y), step=l_step)
        encoder_loss = -(rate + variational_t + variational_y)
        return encoder_loss


def test_sylvester_flow():
    flow = SylvesterFlow(
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
        householder_maps=8,
        name_tag="Sylvester flow",
        feature_maps=256,
        architecture_type="FC_net",
        log_steps=10,
        flow_type_variational="PlanarFlow",
        debug=False
    )


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')
    test_sylvester_flow()
