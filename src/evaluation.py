import numpy as np
import tensorflow as tf


def calc_stats(model, dataset, params):
    """ Function for evaluating a model according to our metrics

   
    Function that calls all individual metrics that we wat to computes 
        and returns a tuple of all metrics.

        Args: 
            ypred1, ypred0: output of get_y0_y1, rescaled with original std and mean of y
    """

    def rmse_ite(ypred1, ypred0):
        pred_ite = tf.zeros_like(true_ite)
        t_new = tf.squeeze(t)
        idx1, idx0 = tf.where(t_new == 1), tf.where(t_new == 0)
        ite1 = tf.gather_nd(y, idx1) - tf.gather_nd(ypred0, idx1)
        ite0 = tf.gather_nd(ypred1, idx0) - tf.gather_nd(y, idx0)
        pred_ite += tf.scatter_nd(idx1, ite1, pred_ite.shape)
        pred_ite += tf.scatter_nd(idx0, ite0, pred_ite.shape)
        return tf.sqrt(tf.reduce_mean(tf.square(true_ite - pred_ite)))

    def abs_ate(ypred1, ypred0):
        return tf.abs(tf.reduce_mean(ypred1 - ypred0) - tf.reduce_mean(true_ite))

    def pehe(ypred1, ypred0):
        return tf.sqrt(tf.reduce_mean(tf.square((mu_1 - mu_0) - (ypred1 - ypred0))))

    def y_errors(y0, y1):
        ypred = (1 - t) * y0 + t * y1
        ypred_cf = t * y0 + (1 - t) * y1
        return y_errors_pcf(ypred, ypred_cf)

    def y_errors_pcf(ypred, ypred_cf):
        rmse_factual = tf.sqrt(tf.reduce_mean(tf.square(ypred - y)))
        rmse_cfactual = tf.sqrt(tf.reduce_mean(tf.square(ypred_cf - y_cf)))
        return rmse_factual, rmse_cfactual

    len_dataset = 0
    for _ in dataset:
        len_dataset +=1
    dataset = dataset.shuffle(len_dataset)

    ite_vec = []
    ate_vec = []
    pehe_vec = []

    for i, features in dataset.batch(params["batch_size"]).enumerate(0):
        x_bin, x_cont, t, y, y_cf, mu_0, mu_1 = features
        true_ite = mu_1 - mu_0
        #encoder_params, decoder_params = model(features, step=0, training=False)
        out = model(features, step=0, training=False)

        ypred1, ypred0 = y, y_cf

        ite_vec += [rmse_ite(ypred1, ypred0)]
        ate_vec += [abs_ate(ypred1, ypred0)]
        pehe_vec += [pehe(ypred1, ypred0)]

    # Rescale last value for having a smaller batch size
    remainer_ratio = features[0].shape[0] / params["batch_size"]
    ite_vec[-1] /= remainer_ratio
    ate_vec[-1] /= remainer_ratio
    pehe_vec[-1] /= remainer_ratio
 
    return tf.reduce_mean(ite_vec), tf.reduce_mean(ate_vec), tf.reduce_mean(pehe_vec)
