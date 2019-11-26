import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.activations import softplus
from tensorflow_probability import distributions as tfd

from fc_net import FC_net
from dataset import IHDP_dataset


class CEVAE(Model):

    def __init__(self, x_bin_size, x_cont_size, z_size, hidden_size=64, debug=False):
        super().__init__()
        """ CEVAE model with fc nets between random variables.
        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py

        After some attempts to port the example code 1 to 1 to TF2, I decided to restructure the
        encoder and decoder to a Model subclass instead to two functions.

        Several fc_nets have an output size of *2 something. This is to output both the mean and 
        std of a Normal distribution at once.
        """
        self.x_bin_size = x_bin_size 
        self.x_cont_size = x_cont_size 
        x_size = x_bin_size + x_cont_size
        self.z_size = z_size
        self.debug = debug

        self.latent_prior = tfd.Independent(tfd.Normal(
                                                loc=tf.zeros([1, z_size], dtype=tf.float64),
                                                scale=tf.ones([1, z_size], dtype=tf.float64)),
                                   reinterpreted_batch_ndims=1)


        # Encoder part
        self.qt_logits = FC_net(x_size, 1, "qt", hidden_size=hidden_size, debug=debug)
        self.hqy       = FC_net(x_size, hidden_size, "hqy", hidden_size=hidden_size, debug=debug)
        self.mu_qy_t0  = FC_net(hidden_size, 1, "mu_qy_t0", hidden_size=hidden_size, debug=debug)
        self.mu_qy_t1  = FC_net(hidden_size, 1, "mu_qy_t1", hidden_size=hidden_size, debug=debug)

        self.hqz   = FC_net(x_size + 1, hidden_size, "hqz", hidden_size=hidden_size, debug=debug)
        self.qz_t0 = FC_net(hidden_size, z_size * 2, "qz_t0", hidden_size=hidden_size, debug=debug) 
        self.qz_t1 = FC_net(hidden_size, z_size * 2, "qz_t1", hidden_size=hidden_size, debug=debug) 

        # Decoder part
        self.hx            = FC_net(z_size, hidden_size, "hx", hidden_size=hidden_size, debug=debug)
        self.x_cont_logits = FC_net(hidden_size, x_cont_size * 2, "x_cont", hidden_size=hidden_size, debug=debug)
        self.x_bin_logits  = FC_net(hidden_size, x_bin_size, "x_bin", hidden_size=hidden_size, debug=debug)

        self.t_logits = FC_net(z_size, 1, "t", hidden_size=hidden_size, debug=debug)
        self.mu_y_t0 = FC_net(z_size, 1, "mu_y_t0", hidden_size=hidden_size, debug=debug)
        self.mu_y_t1 = FC_net(z_size, 1, "mu_y_t1", hidden_size=hidden_size, debug=debug)


    def encode(self, x, t, y, step):
        if self.debug:
            print("Encoding")
        qt = tfd.Independent(tfd.Bernoulli(logits=self.qt_logits(x, step)), 
                             reinterpreted_batch_ndims=1,
                             name="qt")
        qt_sample = tf.dtypes.cast(qt.sample(), tf.float64)
        
        hqy = self.hqy(x, step)
        mu_qy0 = self.mu_qy_t0(hqy, step)
        mu_qy1 = self.mu_qy_t1(hqy, step)
        qy = tfd.Independent(tfd.Normal(loc=qt_sample * mu_qy1 + (1. - qt_sample) * mu_qy0, 
                                        scale=tf.ones_like(mu_qy0)),
                             reinterpreted_batch_ndims=1,
                             name="qy")
        
        xy = tf.concat([x, qy.sample()], 1)
        hidden_z = self.hqz(xy, step)
        qz0 = self.qz_t0(hidden_z, step)
        qz1 = self.qz_t1(hidden_z, step)
        qz = tfd.Independent(tfd.Normal(loc=qt_sample * qz1[:, :self.z_size] + 
                                            (1. - qt_sample) * qz0[:, :self.z_size], 
                                        scale=qt_sample * softplus(qz1[:, self.z_size:]) + 
                                              (1. - qt_sample) * softplus(qz0[:, self.z_size:]),
                                       ),
                             reinterpreted_batch_ndims=1,
                             name="qz")
        return qt, qy, qz

    def decode(self, z, step):
        if self.debug:
            print("Decoding")
        hidden_x = self.hx(z, step)
        x_bin = tfd.Independent(tfd.Bernoulli(logits=self.x_bin_logits(hidden_x, step)),
                                reinterpreted_batch_ndims=1,
                                name="x_bin")

        x_cont_l = self.x_cont_logits(hidden_x, step)
        x_cont = tfd.Independent(tfd.Normal(loc=x_cont_l[:, :self.x_cont_size], 
                                            scale=softplus(x_cont_l[:, self.x_cont_size:])),
                                 reinterpreted_batch_ndims=1,
                                 name="x_cont")

        t = tfd.Independent(tfd.Bernoulli(logits=self.t_logits(z, step)),
                            reinterpreted_batch_ndims=1,
                            name="t")
        t_sample = tf.dtypes.cast(t.sample(), tf.float64)

        mu_y0 = self.mu_y_t0(z, step)
        mu_y1 = self.mu_y_t1(z, step)
        y = tfd.Independent(tfd.Normal(loc=t_sample * mu_y1 + (1. - t_sample) * mu_y0, 
                                       scale=tf.ones_like(mu_y0)),
                            reinterpreted_batch_ndims=1,
                            name="y")
        return x_bin, x_cont, t, y

    @tf.function
    def call(self, features, step, training=False):
        if self.debug:
            print("Starting forward pass")
        x_bin, x_cont, t, y, y_cf, mu_0, mu_1 = features
        x = tf.concat([x_bin, x_cont], 1)

        qt, qy, qz = self.encode(x, t, y, step)
        qz_sample = qz.sample()
        x_bin_likelihood, x_cont_likelihood, t_likelihood, y_likelihood = self.decode(qz_sample, step)

        # Reconstruction
        if self.debug:
            print("Calculating negative data log likelihood")
        distortion_x = -x_bin_likelihood.log_prob(x_bin) - x_cont_likelihood.log_prob(x_cont)
        distortion_t = -t_likelihood.log_prob(t)
        distortion_y = -y_likelihood.log_prob(y)
        avg_x_distortion = tf.reduce_mean(input_tensor=distortion_x)
        avg_t_distortion = tf.reduce_mean(input_tensor=distortion_t)
        avg_y_distortion = tf.reduce_mean(input_tensor=distortion_y)
        tf.summary.scalar("distortion/x", avg_x_distortion, step=step)
        tf.summary.scalar("distortion/t", avg_t_distortion, step=step)
        tf.summary.scalar("distortion/y", avg_y_distortion, step=step)

        # KL-divergence
        if self.debug:
            print("Calculating KL-divergence")
        rate = tfd.kl_divergence(qz, self.latent_prior)
        avg_rate = tf.reduce_mean(input_tensor=rate)
        tf.summary.scalar("rate/z", avg_rate, step=step)

        # Auxillary distributions
        if self.debug:
            print("Calculating negative log likelihood of auxillary distributions")
        variational_t = -qt.log_prob(t)
        variational_y = -qy.log_prob(y)
        avg_variational_t = tf.reduce_mean(variational_t)
        avg_variational_y = tf.reduce_mean(variational_y)
        tf.summary.scalar("variational_ll/t", avg_variational_t, step=step)
        tf.summary.scalar("variational_ll/y", avg_variational_y, step=step)

        return distortion_x, distortion_t, distortion_y, rate, variational_t, variational_y

    @tf.function
    def elbo(self, distortion_x, distortion_t, distortion_y, rate, variational_t, variational_y):
        if self.debug:
            print("Calculating loss")
        elbo_local = -(rate + distortion_x + distortion_t + distortion_y + variational_t + variational_y)
        elbo = tf.reduce_mean(input_tensor=elbo_local)
        return -elbo

    def grad(self, features, _, step):
        with tf.GradientTape() as tape:
            model_output = self(features, step)
            loss = self.elbo(*model_output)
            tf.summary.scalar("metrics/loss", loss, step=step)
        if self.debug:
            print(f"Forward pass complete, step: {step}")
        return loss, tape.gradient(loss, self.trainable_variables)

def train_cevae(params):
    params["x_bin_size"] = 19
    params["x_cont_size"] = 6
    params["z_size"] = 16

    if params["dataset"] == "IHDP":
        dataset = IHDP_dataset(batch_size=params["batch_size"])
    len_dataset = 0
    for _ in dataset:
        len_dataset +=1

    timestamp = str(int(time.time()))[2:] # Cut off the first two digits because this project wont take more than ten years
    logdir = f"{params['model_dir']}cevae/{params['dataset']}/{params['learning_rate']}/{timestamp}"
    if not params["debug"]:
        writer = tf.summary.create_file_writer(logdir)


    cevae = CEVAE(params["x_bin_size"], 
                  params["x_cont_size"], 
                  params["z_size"],
                  debug=params["debug"])
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])

    tf.summary.trace_on(graph=True, profiler=False)

    if params["debug"]:
        for epoch in range(5):
            print(f"Epoch: {epoch}")
            step_start = epoch * (len_dataset // params["batch_size"] + 1)
            for step, features in dataset.batch(params["batch_size"]).enumerate(step_start):
                loss_value, grads = cevae.grad(*features, step)
                #tf.summary.trace_export(name="test?", step=step, profiler_outdir=logdir)
                optimizer.apply_gradients(zip(grads, cevae.trainable_variables))
                if step == 0:
                    tf.summary.trace_export("Profile")
            print("Epoch done")
        sys.exit(0)

    with writer.as_default():
        for epoch in range(params["epochs"]):
            print(f"Epoch: {epoch}")
            step_start = epoch * (len_dataset // params["batch_size"] + 1)
            dataset.shuffle(len_dataset)
            for step, features in dataset.batch(params["batch_size"]).enumerate(step_start):
                loss_value, grads = cevae.grad(*features, step)
                optimizer.apply_gradients(zip(grads, cevae.trainable_variables))
                if step == 0:
                    tf.summary.trace_export("Profile", step=step)


