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
        t_loss=None,
        y_loss=None,
        name_tag="no_name",
        debug=False,
        **kwargs
    ):
        """
        Parameters
        ----------

        """

        super().__init__(name=name_tag)
        self.debug = debug
        self.log_steps = 10
        self.annealing_factor = 1.
        self.t_loss = t_loss
        self.y_loss = y_loss

    @tf.function
    def loss(self, features, encoder_params, step):
        x_cat, x_cont, t, _, y, *_ = features
        qt_prob, qy_mean, qz_mean, qz_std = encoder_params
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
        encoder_loss = -(rate * self.annealing_factor + variational_t +
                         variational_y)
        return encoder_loss


class EncoderCategorical(Encoder):
    """ Encoder subclass for categorical interventions.

    """

    def __init__(
        self,
        x_dims=30,
        t_dims=2,
        y_dims=1,
        y_dist=None,
        y_loss=None,
        y_activation=None,
        z_dims=32,
        name_tag="no_name",
        feature_maps=200,
        architecture_type="FC_net",
        debug=False,
    ):
        """
        Parameters
        ----------

        """

        super().__init__(
            t_loss=CategoricalCrossentropy(),
            y_loss=y_loss,
            name_tag=name_tag,
            debug=debug,
        )
        self.t_dims = t_dims
        self.y_dims = y_dims
        self.y_dist = y_dist
        self.y_activation = y_activation
        self.z_dims = z_dims
        network = eval(architecture_type)

        if architecture_type == "ResNet":
            intermediate_dims = feature_maps * 2
        else:
            intermediate_dims = feature_maps // 2

        self.qt_logits = network(in_dims=x_dims, out_dims=x_dims,
                                 name_tag="qt", n_layers=1,
                                 feature_maps=feature_maps,
                                 squeeze=True, squeeze_dims=t_dims,
                                 debug=debug)

        self.hqy = network(in_dims=x_dims, out_dims=x_dims,
                           name_tag="hqy", n_layers=3,
                           feature_maps=feature_maps, squeeze=True,
                           squeeze_dims=intermediate_dims, debug=debug)

        self.mu_qy_t = FC_net(in_dims=intermediate_dims,
                              out_dims=y_dims * t_dims,
                              name_tag="mu_qy_t", n_layers=2,
                              feature_maps=feature_maps, debug=debug)

        self.qz_t = FC_net(in_dims=intermediate_dims + y_dims,
                           out_dims=z_dims * 2 * t_dims,
                           name_tag="qz_t", n_layers=3,
                           feature_maps=feature_maps, debug=debug)

    @tf.function
    def call(self, x, t, y, step, training=False):
        if self.debug:
            print("Encoding")

        qt_prob = nn.softmax(self.qt_logits(x, step, training=training))
        qt = tfd.Independent(tfd.OneHotCategorical(probs=qt_prob,
                                                   dtype=tf.float64),
                             reinterpreted_batch_ndims=1,
                             name='qt')
        qt_sample = qt.sample()

        hqy = self.hqy(x, step, training=training)
        mu_qy_t = self.mu_qy_t(hqy, step, training=training)
        mu_qy_t = tf.reshape(mu_qy_t, (len(x), self.y_dims, self.t_dims))
        shape = tf.concat([[len(x)],
                           tf.ones(tf.rank(mu_qy_t) - 2, dtype=tf.int32),
                           [qt_sample.shape[-1]]], axis=0)

        if training:
            t = tf.reshape(t, shape)
            qy_mean = tf.reduce_sum(t * mu_qy_t, axis=-1)
        else:
            qt_sample = tf.reshape(qt_sample, shape)
            qy_mean = tf.reduce_sum(qt_sample * mu_qy_t, axis=-1)
        qy_mean = self.y_activation(qy_mean)
        qy = tfd.Independent(self.y_dist(qy_mean),
                             reinterpreted_batch_ndims=1,
                             name='qy')

        if training:
            xy = tf.concat([hqy, y], -1)
        else:
            xy = tf.concat([hqy, qy.sample()], -1)

        qz_t = self.qz_t(xy, step, training=training)
        qz_t = tf.reshape(qz_t, (len(x), self.z_dims * 2, self.t_dims))
        qz_mean, qz_std = tf.split(qz_t, 2, axis=-2)

        if training:
            qz_mean = tf.reduce_sum(t * qz_mean, axis=-1)
            qz_std = tf.reduce_sum(t * softplus(qz_std), axis=-1)
        else:
            qz_mean = tf.reduce_sum(qt_sample * qz_mean, axis=-1)
            qz_std = tf.reduce_sum(qt_sample * softplus(qz_std), axis=-1)
        return qt_prob, qy_mean, qz_mean, qz_std


class EncoderContinuous(Encoder):
    """ Encoder subclass for continuous interventions.

    """

    def __init__(
        self,
        x_dims=30,
        t_dims=2,
        y_dims=1,
        y_dist=None,
        y_loss=None,
        y_activation=None,
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

        super().__init__(
            t_loss=MeanSquaredError(),
            y_loss=y_loss,
            name_tag=name_tag,
            debug=debug
        )

        self.y_dist = y_dist
        self.y_activation = y_activation
        self.z_dims = z_dims
        network = eval(architecture_type)

        if architecture_type == "ResNet":
            intermediate_dims = feature_maps * 2
        else:
            intermediate_dims = feature_maps // 2

        self.qt_logits = network(in_dims=x_dims, out_dims=x_dims,
                                 name_tag="qt", n_layers=1,
                                 feature_maps=feature_maps,
                                 squeeze=True, squeeze_dims=t_dims,
                                 debug=debug)

        self.hqy = network(in_dims=x_dims, out_dims=x_dims,
                           name_tag="hqy", n_layers=3,
                           feature_maps=feature_maps, squeeze=True,
                           squeeze_dims=intermediate_dims, debug=debug)

        self.mu_qy_t = FC_net(in_dims=intermediate_dims + t_dims,
                              out_dims=y_dims,
                              name_tag="mu_qy_t", n_layers=2,
                              feature_maps=feature_maps, debug=debug)

        self.qz_t = FC_net(in_dims=intermediate_dims + y_dims + t_dims,
                           out_dims=z_dims * 2,
                           name_tag="qz_t", n_layers=3,
                           feature_maps=feature_maps, debug=debug)

    @tf.function
    def call(self, x, t, y, step, training=False):
        if self.debug:
            print("Encoding")
        qt_prob = self.qt_logits(x, step, training=training)
        qt = tfd.Independent(tfd.Normal(qt_prob, scale=tf.ones_like(qt_prob)),
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

        return qt_prob, qy_mean, qz_mean, qz_std
