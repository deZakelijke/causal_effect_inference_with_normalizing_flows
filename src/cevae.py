import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.activations import softplus
from tensorflow_probability import distributions as tfd

from fc_net import FC_net
from dataset import IHDP_dataset


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
    t_logits = FC_net(1, hidden_size)
    mu_y_t0 = FC_net(1, hidden_size)
    mu_y_t1 = FC_net(1, hidden_size)
    #y       = tfd.Normal(loc=t * mu_y_t1 + (1. - t) * mu_y_t0, scale=tf.ones_like(mu_y_t0))


    def encode(x, t, y):
        qt = tfd.Independent(tfd.Bernoulli(logits=qt_logits(x)), 
                             reinterpreted_batch_ndims=1,
                             name="qt").sample()
        qt = tf.dtypes.cast(qt, tf.float64)
        mu_qy0 = mu_qy_t0(hqy(qt))
        mu_qy1 = mu_qy_t1(hqy(qt))
        qy = tfd.Independent(tfd.Normal(loc=qt * mu_qy1 + (1. - qt) * mu_qy0, 
                                        scale=tf.ones_like(mu_qy0)),
                             reinterpreted_batch_ndims=2,
                             name="qy").sample()
        
        xy = tf.concat([x, qy], 1)
        hidden_z = hqz(xy)
        qz0 = qz_t0(hidden_z)
        qz1 = qz_t1(hidden_z)
        qz = tfd.Independent(tfd.Normal(loc=qt * qz1[:, :z_size] + (1. - qt) * qz0[:, :z_size], 
                                        scale=qt * softplus(qz1[:, z_size:]) + (1. - qt) * 
                                        softplus(qz0[:, z_size:]),
                                        #DType=tf.float64
                                       ),
                             reinterpreted_batch_ndims=2,
                             name="qz")
        return qt, qy, qz

    def decode(z):

        hidden_x = hx(z)
        x_bin = tfd.Independent(tfd.Bernoulli(logits=x_bin_logits(hidden_x)),
                                reinterpreted_batch_ndims=2,
                                name="x_bin").sample()
        x_bin = tf.dtypes.cast(x_bin, tf.float64)
        x_cont_l = x_cont_logits(hidden_x)
        x_cont = tfd.Independent(tfd.Normal(loc=x_cont_l[:, :x_cont_size], 
                                            scale=softplus(x_cont_l[:, x_cont_size:])),
                                 reinterpreted_batch_ndims=2,
                                 name="x_cont").sample()
        x = tf.concat([x_bin, x_cont], 1)

        t = tfd.Independent(tfd.Bernoulli(logits=t_logits(z)),
                            reinterpreted_batch_ndims=2,
                            name="t").sample()
        t = tf.dtypes.cast(t, tf.float64)

        mu_y0 = mu_y_t0(z)
        mu_y1 = mu_y_t1(z)
        y = tfd.Independent(tfd.Normal(loc=t * mu_y1 + (1. - t) * mu_y0, 
                                       scale=tf.ones_like(mu_y0)),
                            reinterpreted_batch_ndims=2,
                            name="y")
        return x, t, y


    return encode, decode

def model_fn(features, labels, mode, params, config):
    """Builds the model function for use in an estimator.

    Arguments:
        features: The input features for the estimator.
        labels: The labels, unused here.
        mode: Signifies whether it is train or test or predict.
        params: Some hyperparameters as a dictionary.
        config: The RunConfig, unused here.

    Returns:
        EstimatorSpec: A tf.estimator.EstimatorSpec instance.
    """
    del labels, config

    encoder, decoder = make_cevae(params["x_bin_size"], 
                                  params["x_cont_size"], 
                                  params["z_size"])
    latent_prior = tfd.Independent(tfd.Normal(loc=tf.zeros([1, params["z_size"]], dtype=tf.float64),
                                        scale=tf.ones([1, params["z_size"]], dtype=tf.float64)),
                                   reinterpreted_batch_ndims=2)


    x_bin, x_cont, t, y, y_cf, mu_0, mu_1 = features
    x = tf.concat([x_bin, x_cont], 1)

    qt, qy, qz = encoder(x, t, y)
    qz_sample = qz.sample() # number of samples of z
    x_likelihood, t_likelihood, y_likelihood = decoder(qz_sample)

    # Reconstruction - incomplete
    distortion_y = -y_likelihood.log_prob(y)
    avg_y_distortion = tf.reduce_mean(input_tensor=distortion_y)
    tf.summary.scalar("distortion_y", avg_y_distortion)

    # KL-divergence - incomplete
    rate = tfd.kl_divergence(qz, latent_prior)
    avg_rate = tf.reduce_mean(input_tensor=rate)
    tf.summary.scalar("rate_z", avg_rate)

    # Total elbo
    elbo_local = -(rate + distortion_y)
    elbo = tf.reduce_mean(input_tensor=elbo_local)
    loss = -elbo
    tf.summary.scalar("elbo", elbo)

    # Learning
    #global_step = tf.train.get_or_create_global_step()
    #global_step = tf.Variable(1, name="global_step")
    #learning_rate = tf.train.cosine_decay(params["learning_rate"], 
    #                                      global_step,
    #                                      params["max_steps"])
    #tf.summary.scalar("learning_rate", learning_rate)
    optimizer = tf.optimizers.Adam(params["learning_rate"])
    train_op = optimizer.minimize(loss)


    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={
            "elbo": tf.metrics.mean(elbo),
            "rate_z": tf.metrics.mean(avg_rate),
            "distortion_y": tf.metrics.mean(distortion_y),
        },
    )


def train_cevae(params):
    params["x_bin_size"] = 10
    params["x_cont_size"] = 10
    params["z_size"] = 64

    if params["dataset"] == "IHDP":
        dataset_fn = IHDP_dataset(batch_size=params["batch_size"])

    #writer = tf.summary.create_file_writer(params["model_dir"])

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=tf.estimator.RunConfig(
            model_dir=params["model_dir"],
            save_checkpoints_steps=params["save_steps"],
        ),
    )

    for i in range(10):
        estimator.train(dataset_fn, steps=10)


