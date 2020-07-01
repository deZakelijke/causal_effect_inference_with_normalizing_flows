import numpy as np
import tensorflow as tf


def calc_stats(model, dataset, scaling_data, params):
    """ Function for evaluating a model according to our metrics


    Function that calls all individual metrics that we wat to computes
    and returns a tuple of all metrics.

    Parameters
    ----------
    model : tf.keras.Model
        The model that does the predictions

    dataset : tf.data.Dataset

    scaling_data : (float, float)

    params : dict
    """

    nr_samples = 100

    def rmse_ite(ypred1, ypred0, y):
        # pred_ite = tf.zeros_like(true_ite)
        pred_ite = tf.zeros((len(true_ite), 1), dtype=tf.float64)
        t_new = tf.squeeze(t)
        idx1, idx0 = tf.where(t_new == 1), tf.where(t_new == 0)
        ite1 = tf.gather_nd(y, idx1) - tf.gather_nd(ypred0, idx1)
        ite0 = tf.gather_nd(ypred1, idx0) - tf.gather_nd(y, idx0)
        pred_ite += tf.scatter_nd(idx1, ite1, pred_ite.shape)
        pred_ite += tf.scatter_nd(idx0, ite0, pred_ite.shape)
        return tf.square(true_ite - pred_ite)

    def abs_ate(ypred1, ypred0, true_ite):
        """
         This would average out different effects if the treatment would
         work differently on different on different parts of the population.
        """
        return tf.concat([ypred1 - ypred0, true_ite], -1)

    def pehe(ypred1, ypred0, mu_1, mu_0):
        return tf.square((mu_1 - mu_0) - (ypred1 - ypred0))

    def y_errors(ypred0, ypred1, y, y_cf):
        ypred = (1 - t) * ypred0 + t * ypred1
        ypred_cf = t * ypred0 + (1 - t) * ypred1
        se_factual = tf.square(ypred - y)
        se_cfactual = tf.square(ypred_cf - y_cf)
        return tf.concat([se_factual, se_cfactual], -1)

    """
    def y_errors_pcf(ypred, ypred_cf, y, y_cf):
        rmse_factual = tf.sqrt(tf.reduce_mean(tf.square(ypred - y)))
        rmse_cfactual = tf.sqrt(tf.reduce_mean(tf.square(ypred_cf - y_cf)))
        return rmse_factual, rmse_cfactual
    """

    len_dataset = 0
    for _ in dataset:
        len_dataset += 1
    dataset = dataset.shuffle(len_dataset)

    ite_scores = tf.Variable(tf.zeros((len_dataset, 1), dtype=tf.double))
    ate_scores = tf.Variable(tf.zeros((len_dataset, 2), dtype=tf.double))
    pehe_scores = tf.Variable(tf.zeros((len_dataset, 1), dtype=tf.double))
    y_error_val = tf.Variable(tf.zeros((len_dataset, 2), dtype=tf.double))

    for i, features in dataset.batch(params["batch_size"]).enumerate(0):
        x_bin, x_cont, t, t_cf, y, y_cf, mu_0, mu_1 = features
        t = tf.expand_dims(t[:, 0], axis=-1)
        t_cf = tf.expand_dims(t_cf[:, 0], axis=-1)
        # Why were we doing this? Because it was a binary one-hot encoding
        # So we want to both generalise our metric and generalise the method
        # That does the intervention. Both assume a binary intervention. 
        # The intervention method could be changed to factual and counterfactual
        # but what should the value of the counterfactual intervention be?
        true_ite = mu_1 - mu_0

        x = tf.concat([x_bin, x_cont], -1)
        ypred0, ypred1 = model.do_intervention(x, tf.zeros_like(t),
                                               tf.ones_like(t_cf), nr_samples)
        # ypred0 = tf.expand_dims(ypred0, -1)
        # ypred1 = tf.expand_dims(ypred1, -1)
        y_mean, y_std = scaling_data[0], scaling_data[1]
        ypred0, ypred1 = ypred0 * y_std + y_mean, ypred1 * y_std + y_mean
        y = y * y_std + y_mean
        y_cf = y_cf * y_std + y_mean

        slice_indices = (i * features[0].shape[0],
                         (i + 1) * features[0].shape[0])
        ite_scores = ite_scores[slice_indices[0]:slice_indices[1]].\
            assign(rmse_ite(ypred1, ypred0, y))
        ate_scores = ate_scores[slice_indices[0]:slice_indices[1]].\
            assign(abs_ate(ypred1, ypred0, true_ite))
        pehe_scores = pehe_scores[slice_indices[0]:slice_indices[1]].\
            assign(pehe(ypred1, ypred0, mu_1, mu_0))
        y_error_val = y_error_val[slice_indices[0]:slice_indices[1]].\
            assign(y_errors(ypred1, ypred0, y, y_cf))

    ite = tf.sqrt(tf.reduce_mean(ite_scores))
    ate = tf.abs(tf.reduce_mean(ate_scores[:, 0]) -
                 tf.reduce_mean(ate_scores[:, 1]))
    pehe = tf.sqrt(tf.reduce_mean(pehe_scores))
    y_rmse = tf.sqrt(tf.reduce_mean(y_error_val, axis=1))

    return ite, ate, pehe, y_rmse
