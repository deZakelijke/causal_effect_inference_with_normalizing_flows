import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.activations import softplus
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow import nn
from tensorflow_probability import distributions as tfd

from decoder import DecoderCategorical, DecoderContinuous
from encoder import EncoderCategorical, EncoderContinuous
from fc_net import FC_net
from res_net import ResNet
from utils import get_log_prob, get_analytical_KL_divergence


class CEVAE(Model):
    """ CEVAE model with fc nets between random variables.
    https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py

    After some attempts to port the example code 1 to 1 to TF2, I decided
    to restructure the encoder and decoder to a Model subclass instead to
    two functions.

    Several fc_nets have an output size of *2 something. This is to output
    both the mean and std of a Normal distribution at once.
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
        super().__init__(name=name_tag)
        self.debug = debug
        # self.category_sizes = category_sizes
        # self.t_dims = t_dims
        # self.y_dims = y_dims
        # self.log_steps = log_steps
        # self.architecture_type = architecture_type

        # self.annealing_factor = 1e-4
        self.annealing_factor = 1.

        if architecture_type == "ResNet":
            self.x_cat_dims = x_cat_dims
        else:
            self.x_cat_dims = (x_cat_dims, )

        if y_type == "Categorical":
            y_loss = CategoricalCrossentropy()
            y_dist = lambda x: tfd.OneHotCategorical(probs=x,
                                                     dtype=tf.float64)
            y_activation = nn.softmax
        else:
            y_loss = MeanSquaredError()
            y_dist = lambda x: tfd.Normal(x, scale=tf.ones_like(x))
            y_activation = lambda x: x

        if t_type == "Categorical":
            self.encoder = EncoderCategorical(x_dims, t_dims, y_dims, y_dist,
                                              y_loss, y_activation, z_dims,
                                              "Encoder", feature_maps,
                                              architecture_type, debug)
            self.decoder = DecoderCategorical(x_cat_dims, x_cont_dims, t_dims,
                                              y_dims, y_loss, y_activation,
                                              z_dims, category_sizes,
                                              "Decoder", feature_maps,
                                              architecture_type, debug)
        else:

            self.encoder = EncoderContinuous(x_dims, t_dims, y_dims, y_dist,
                                             y_loss, y_activation, z_dims,
                                             "Encoder", feature_maps,
                                             architecture_type, debug)
            self.decoder = DecoderContinuous(x_cat_dims, x_cont_dims, t_dims,
                                             y_dims, y_loss, y_activation,
                                             z_dims, category_sizes, "Decoder",
                                             feature_maps, architecture_type,
                                             debug)

    @tf.function
    def call(self, x, t, y, step, training=False):
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

        encoder_params = self.encoder(x, t, y, step, training=training)
        _, _, qz_mean, qz_std = encoder_params
        qz = tf.random.normal(qz_mean.shape, dtype=tf.float64)
        qz = qz * qz_std + qz_mean

        decoder_params = self.decoder(qz, t, step, training=training)
        return encoder_params, decoder_params

    @tf.function
    def loss(self, features, encoder_params, decoder_params, step):
        if self.debug:
            print("Calculating loss")
        encoder_loss = self.encoder.loss(features, encoder_params, step)
        decoder_loss = self.decoder.loss(features, decoder_params, step)

        elbo_local = encoder_loss + decoder_loss
        elbo = tf.reduce_mean(elbo_local)
        return -elbo

    def do_intervention(self, x, t0, t1, n_samples):
        """ Perform two interventions to compare downstream.

        Use n_samples for both number of samples from latent space
        and for number of samples from intervention distribution.

        """
        # Get latent confounder
        *_, qz_mean, qz_std = self.encoder(x, None, None, None, training=False)
        # final_shape = (n_samples, qz_mean.shape[0], self.y_dims, self.t_dims)
        qz = tf.random.normal((n_samples, *qz_mean.shape), dtype=tf.float64)
        z = qz * qz_std + qz_mean

        # Do simulation with intervention variable
        # We have to sample t0 and t1 several times
        y0, y1 = self.decoder.do_intervention(z, t0, t1, n_samples)
        return y0, y1
