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


class Decoder(Model):

    def __init__(
        self,
        # x_cat_dims=10,
        t_dims=2,
        t_loss=None,
        y_dims=1,
        y_loss=None,
        category_sizes=2,
        name_tag="no_name",
        debug=False,
    ):
        """
        Parameters
        ----------
    
        """
    
        super().__init__(name=name_tag)
        self.category_sizes = category_sizes
        self.t_dims = t_dims
        self.t_loss = t_loss
        # self.t_activation = t_activation
        self.y_dims = y_dims
        self.y_loss = y_loss
        # self.y_activation = y_activation
        # self.z_dims = z_dims
        self.log_steps = 10
        self.debug = debug
    
    @tf.function
    def loss(self, features, decoder_params, step):
        x_cat, x_cont, t, _, y, *_ = features
        x_cat_prob, x_cont_mean, x_cont_std, t_prob, y_mean = decoder_params
        x_cat = tf.reshape(x_cat, (len(x_cat), *self.x_cat_dims,
                                   self.category_sizes))
    
        distortion_x_cat = CategoricalCrossentropy()(x_cat, x_cat_prob)
        # Use relu to set nans to zero
        distortion_x_cont = nn.relu(MeanSquaredError()(x_cont, x_cont_mean))
        distortion_x = distortion_x_cat + distortion_x_cont
        distortion_t = self.t_loss(t, t_prob)
        distortion_y = self.y_loss(y, y_mean)
        if step is not None and step % (self.log_steps * 10) == 0:
            l_step = step // (self.log_steps * 10)
            if self.architecture_type == "ResNet":
                with tf.device("cpu:0"):
                    tf.summary.image("reconstructed_proxy",
                                     x_cont_mean, step=l_step, max_outputs=4)
            tf.summary.scalar("partial_loss/distortion_x",
                              tf.reduce_mean(distortion_x), step=l_step)
            tf.summary.scalar("partial_loss/distortion_x_cat",
                              tf.reduce_mean(distortion_x_cat), step=l_step)
            tf.summary.scalar("partial_loss/distortion_x_cont",
                              tf.reduce_mean(distortion_x_cont), step=l_step)
    
            tf.summary.scalar("partial_loss/distortion_t",
                              tf.reduce_mean(distortion_t), step=l_step)
            tf.summary.scalar("partial_loss/distortion_y",
                              tf.reduce_mean(distortion_y), step=l_step)
    
        decoder_loss = -(distortion_x + distortion_t + distortion_y)
        return decoder_loss


class DecoderCategorical(Decoder):

    def __init__(
        self,
        x_cat_dims=10,
        x_cont_dims=10,
        t_dims=2,
        y_dims=1,
        y_loss=None,
        y_activation=None,
        z_dims=32,
        category_sizes=2,
        name_tag="no_name",
        feature_maps=256,
        architecture_type="FC_net",
        debug=False,
        **_
    ):
        super().__init__(
            # x_cat_dims=x_cat_dims,
            t_dims=t_dims,
            t_loss=CategoricalCrossentropy(),
            y_dims=y_dims,
            y_loss=y_loss,
            category_sizes=category_sizes,
            # z_dims=z_dims,
            name_tag=name_tag,
            debug=debug
        )
        self.y_activation = y_activation
        network = eval(architecture_type)
        self.architecture_type = architecture_type
    
        if architecture_type == "ResNet":
            x_out_dims = x_cont_dims[:-1] + (x_cont_dims[-1] * 2 +
                                             x_cat_dims[-1] * category_sizes, )
            self.x_split_dims = [x_cont_dims[-1], x_cont_dims[-1],
                                 x_cat_dims[-1] * category_sizes]
            self.x_cat_dims = x_cat_dims
            intermediate_dims = x_cont_dims[:-1]
        else:
            x_out_dims = x_cont_dims * 2 + x_cat_dims * category_sizes
            self.x_split_dims = [x_cont_dims, x_cont_dims,
                                 x_cat_dims * category_sizes]
            self.x_cat_dims = (x_cat_dims, )
            intermediate_dims = x_cont_dims
    
        self.x_logits = network(in_dims=z_dims, out_dims=x_out_dims,
                                name_tag="x", n_layers=3,
                                feature_maps=feature_maps,
                                unsqueeze=True, squeeze_dims=intermediate_dims,
                                coord_conv=False, debug=debug)
        self.t_logits = FC_net(in_dims=z_dims, out_dims=z_dims, name_tag="t",
                               n_layers=1, feature_maps=feature_maps,
                               squeeze=True, squeeze_dims=t_dims,
                               debug=self.debug)
    
        self.mu_y_t = FC_net(in_dims=z_dims, out_dims=y_dims * t_dims,
                             name_tag="mu_y_t", n_layers=2,
                             feature_maps=feature_maps, debug=debug)

    @tf.function
    def call(self, z, t, step, training=False):
        if self.debug:
            print("Decoding")
        x_logits = self.x_logits(z, step, training=training)
        x_cont_mean, x_cont_logvar, x_cat_logits = tf.split(x_logits,
                                                            self.x_split_dims,
                                                            axis=-1)
        x_cont_std = softplus(x_cont_logvar)
        x_cat_prob = nn.softmax(tf.reshape(x_cat_logits, (len(z),
                                                          *self.x_cat_dims,
                                                          self.category_sizes
                                                          )))
    
        t_prob = nn.softmax(self.t_logits(z, step, training=training))
        mu_y_t = self.mu_y_t(z, step, training=training)
        mu_y_t = tf.reshape(mu_y_t, (len(z), self.y_dims, self.t_dims))
        shape = tf.concat([[len(z)],
                           tf.ones(tf.rank(mu_y_t) - 2, dtype=tf.int32),
                           [t.shape[-1]]], axis=0)
        t = tf.reshape(t, shape)
        y_mean = tf.reduce_sum(t * mu_y_t, axis=-1)
        y_mean = self.y_activation(y_mean)
    
        return x_cat_prob, x_cont_mean, x_cont_std, t_prob, y_mean

    def do_intervention(self, z, t0, t1, n_samples):
        """ Computes the quantity E[y|z, do(t=0)] and E[y|z, do(t=1)]
    
        nr_sample of samples are drawn from the Normal distribution
        N(qz_mean, qz_std) and used to infer y for both t=0 and t=1.
        The samples are averaged at the end.
    
        """
        # in1 = tf.concat([z, tf.tile(tf.expand_dims(t1, 0),
                                    # [n_samples, 1, 1])], -1)
        t0 = tf.tile(tf.expand_dims(t0, 0), [n_samples, 1, 1])
        t1 = tf.tile(tf.expand_dims(t1, 0), [n_samples, 1, 1])

        y = self.mu_y_t(z, None, training=False)
        y = tf.reshape(y, (n_samples, y.shape[1], self.y_dims, self.t_dims))
        shape = tf.concat([[n_samples, y.shape[1]],
                           tf.ones(tf.rank(y) - 3, dtype=tf.int32),
                           [t0.shape[-1]]], axis=0)

        t0 = tf.reshape(t0, shape)
        t1 = tf.reshape(t1, shape)

        y0 = tf.reduce_sum(y * t0, axis=-1)
        y1 = tf.reduce_sum(y * t1, axis=-1)
    
        y0 = tf.reduce_mean(y0, axis=0)
        y1 = tf.reduce_mean(y1, axis=0)
        return y0, y1


class DecoderContinuous(Decoder):


    def __init__(
        self,
        x_cat_dims=10,
        x_cont_dims=10,
        t_dims=2,
        y_dims=1,
        y_loss=None,
        y_activation=None,
        z_dims=32,
        category_sizes=2,
        name_tag="no_name",
        feature_maps=256,
        architecture_type="FC_net",
        debug=False,
        **_
    ):
        super().__init__(
            t_dims=t_dims,
            t_loss=MeanSquaredError(),
            y_dims=y_dims,
            y_loss=y_loss,
            category_sizes=category_sizes,
            name_tag=name_tag,
            debug=debug
        )

        self.y_activation = y_activation
        network = eval(architecture_type)
        self.architecture_type = architecture_type
    
        if architecture_type == "ResNet":
            x_out_dims = x_cont_dims[:-1] + (x_cont_dims[-1] * 2 +
                                             x_cat_dims[-1] * category_sizes, )
            self.x_split_dims = [x_cont_dims[-1], x_cont_dims[-1],
                                 x_cat_dims[-1] * category_sizes]
            self.x_cat_dims = x_cat_dims
            intermediate_dims = x_cont_dims[:-1]
        else:
            x_out_dims = x_cont_dims * 2 + x_cat_dims * category_sizes
            self.x_split_dims = [x_cont_dims, x_cont_dims,
                                 x_cat_dims * category_sizes]
            self.x_cat_dims = (x_cat_dims, )
            intermediate_dims = x_cont_dims
    
        self.x_logits = network(in_dims=z_dims, out_dims=x_out_dims,
                                name_tag="x", n_layers=3,
                                feature_maps=feature_maps,
                                unsqueeze=True, squeeze_dims=intermediate_dims,
                                coord_conv=False, debug=debug)
        self.t_logits = FC_net(in_dims=z_dims, out_dims=z_dims, name_tag="t",
                               n_layers=1, feature_maps=feature_maps,
                               squeeze=True, squeeze_dims=t_dims,
                               debug=self.debug)
    
        self.mu_y_t = FC_net(in_dims=z_dims + t_dims, out_dims=y_dims,
                             name_tag="mu_y_t", n_layers=2,
                             feature_maps=feature_maps, debug=debug)

    @tf.function
    def call(self, z, t, step, training=False):
        if self.debug:
            print("Decoding")
        x_logits = self.x_logits(z, step, training=training)
        x_cont_mean, x_cont_logvar, x_cat_logits = tf.split(x_logits,
                                                            self.x_split_dims,
                                                            axis=-1)
        x_cont_std = softplus(x_cont_logvar)
        x_cat_prob = nn.softmax(tf.reshape(x_cat_logits, (len(z),
                                                          *self.x_cat_dims,
                                                          self.category_sizes
                                                          )))
    
        t_prob = self.t_logits(z, step, training=training)
        y_mean = self.mu_y_t(tf.concat([z, t], -1), step, training=training)
        y_mean = self.y_activation(y_mean)
    
        return x_cat_prob, x_cont_mean, x_cont_std, t_prob, y_mean

    def do_intervention(self, z, t0, t1, n_samples):
        """ Computes the quantity E[y|z, do(t=0)] and E[y|z, do(t=1)]
    
        nr_sample of samples are drawn from the Normal distribution
        N(qz_mean, qz_std) and used to infer y for both t=0 and t=1.
        The samples are averaged at the end.
    
        """
        in0 = tf.concat([z, tf.tile(tf.expand_dims(t0, 0),
                                    [n_samples, 1, 1])], -1)
        in1 = tf.concat([z, tf.tile(tf.expand_dims(t1, 0),
                                    [n_samples, 1, 1])], -1)
        y0 = self.mu_y_t(in0, None, training=False)
        y1 = self.mu_y_t(in1, None, training=False)
    
        y0 = tf.reduce_mean(y0, axis=0)
        y1 = tf.reduce_mean(y1, axis=0)
        return y0, y1
