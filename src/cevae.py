import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.activations import softplus
from tensorflow_probability import distributions as tfd

from fc_net import FC_net
from dataset import IHDP


def make_cevae(x_bin_size, x_cont_size, z_size, hidden_size=512):
    """ CEVAE model with fc nets between random variables.
    https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py
    """

    # Encoder part
    #qt       = tfd.Bernoulli(logits=FC_net(1, hidden_size), dtype=tf.float32)
    qt_logits = FC_net(1, hidden_size)
    hqy      = FC_net(hidden_size, hidden_size)
    mu_qy_t0 = FC_net(1, hidden_size)
    mu_qy_t1 = FC_net(1, hidden_size)
    #qy       = tfd.Normal(loc=qt * mu_qy_t1 + (1. - qt) * mu_qy_t0, scale=tf.ones_like(mu_qy_t0))

    hqz   = FC_net(z_size, hidden_size)
    qz_t0 = FC_net(z_size * 2, hidden_size)
    qz_t1 = FC_net(z_size * 2, hidden_size)
    #qz    = tfd.Normal(loc=self.qt * qz_t1[:, :z_size] + (1. - self.qt) * qz_t0[:, :z_size],
    #        scale=self.qt * softplus(qz_t1[:, z_size:]) + (1. - self.qt) * softplus(qz_t0[:, z_size:]))

    # Decoder part
    #pz = tfd.Normal(loc=tf.zeros([z_size]), scale=tf.ones([z_size]))
    hx      = FC_net(x_bin_size + x_cont_size, hidden_size)
    #x1      = tfd.Bernoulli(logits=FC_net(x_bin_size, hidden_size), dtype=tf.float32)
    x_cont_logits = FC_net(x_cont_size * 2, hidden_size)
    #hx_c    = FC_net(x_cont_size * 2, hidden_size)
    #x2      = tfd.Normal(loc=hx_c[:, :x_cont_size], scale=softplus(hx_c[:, x_cont_size:]))
    x_bin_logits = FC_net(x_bin_size, hidden_size)
    #t       = tdf.Bernoulli(logits=FC_net(1, hidden_size), dtype=tf.float32)
    t_logtis = FC_net(1, hidden_size)
    mu_y_t0 = FC_net(1, hidden_size)
    mu_y_t1 = FC_net(1, hidden_size)
    #y       = tfd.Normal(loc=t * mu_y_t1 + (1. - t) * mu_y_t0, scale=tf.ones_like(mu_y_t0))


    def encode(self, x, y, t):
        qt = tfd.Bernoulli(logits=qt_logits(x))
        mu_qy0 = mu_qy_t0(hqy(qt))
        mu_qy1 = mu_qy_t1(hqy(qt))
        qy = tdf.Normal(loc=qt * mu_qy1 + (1. - qt) * mu_qy0, scale=tf.ones_like(mu0))

        xy = tf.concat([x, qy], 1)
        hidden_z = hqz(xy)
        qz0 = qz_t0(hidden_z)
        qz1 = qz_t1(hidden_z)
        qz = tfd.Normal(loc=qt * qz1[:, :z_size] + (1. - qt) * qz0[:, :z_size], 
                scale=qt * softplus(qz1[:, z_size:]) + (1. - qt) * softplus(qz0[:, z_size:]))
        return qt, qy, qz

    def decode(self, z):
        hidden_x = hx(z)
        x_bin = tfd.Bernoulli(logits=x_bin_logits(hidden_x))
        x_cont_l = x_cont_logits(hidden_x)
        x_cont = tfd.Normal(loc=x_cont_l[:, :x_cont_size], scale=softplus(x_cont_l[:, x_cont_size:]))
        x = (x_bin, x_cont)

        t = tfd.Bernoulli(logits=t_logits(z))

        mu_y0 = mu_y_t0(z)
        mu_y1 = mu_y_t1(z)
        y = tfd.Normal(loc=t * mu_y1 + (1. - t) * mu_y0, scale=tf.ones_like(mu_y0))
        return x, t, y


    return encode, decode



def train_cevae(config):
    x_bin_size = 10
    x_cont_size = 10
    z_size = 64
    encoder, decoder = make_cevae(x_bin_size, x_cont_size, z_size)
    if config.dataset == "IHDP":
        dataset = tf.data.Dataset.from_generator(IHDP, tf.float32)


