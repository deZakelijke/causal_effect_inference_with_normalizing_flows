import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd


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

    n_samples = params['n_samples']

    def rmse_ite(ypred1, ypred0, y, idx0, idx1):
        """ Calculate the square errors of the ITE

        Only the squares of the errors terms are calculated. They are summed
        donwstream.
        It is important to note that we make use of the 'strong ignorability'
        condition to calculate the ITE.
        """
        # pred_ite = tf.zeros_like(true_ite)
        pred_ite = tf.zeros((len(true_ite), 1), dtype=tf.float64)
        # ite1 = tf.gather_nd(y, idx1) - tf.gather_nd(ypred0, idx1)
        ite1 = y * idx1 - ypred0 * idx1
        # ite0 = tf.gather_nd(ypred1, idx0) - tf.gather_nd(y, idx0)
        ite0 = ypred1 * idx0 - y * idx0
        # pred_ite += tf.scatter_nd(idx1, ite1, pred_ite.shape)
        # pred_ite += tf.scatter_nd(idx0, ite0, pred_ite.shape)
        pred_ite = ite1 + ite0
        return tf.square(true_ite - pred_ite)

    def abs_ate(ypred1, ypred0, true_ite):
        """
         This would average out different effects if the treatment would
         work differently on different on different parts of the population.
        """
        return tf.concat([ypred1 - ypred0, true_ite], -1)

    def pehe(ypred1, ypred0, mu_1, mu_0):
        return tf.square((mu_1 - mu_0) - (ypred1 - ypred0))

    def y_errors(ypred0, ypred1, y_predict, idx_f, idx_cf):
        y_err0 = tf.square(y_predict[:, 0] - ypred0)
        y_err1 = tf.square(y_predict[:, 1] - ypred1)
        se_factual = y_err0 * idx_f + y_err1 * idx_cf
        se_cfactual = y_err0 * idx_cf + y_err1 * idx_f
        return tf.concat([se_factual, se_cfactual], -1)

    def calculate_indices(t, t0, t1, y, y0, y1):
        t0_diff = tf.norm(tf.abs(t - t0), axis=1, keepdims=True)
        t1_diff = tf.norm(tf.abs(t - t1), axis=1, keepdims=True)
        y0_diff = tf.norm(tf.abs(y - y0), axis=1, keepdims=True)
        y1_diff = tf.norm(tf.abs(y - y1), axis=1, keepdims=True)

        idx0 = tf.cast(tf.greater(t1_diff, t0_diff), tf.float64)
        idx1 = 1. - idx0

        idx_f = tf.cast(tf.greater(y1_diff, y0_diff), tf.float64)
        idx_cf = 1. - idx_f
        """
        We run into a new problem here, namely that we need to determine
        the indices without exact matches between t and t0, t1.
        Similarly for y and y0, y1
        Since we need to make a split into two pieces we might as well
        just pick the one that is closest in both cases.
        """
        return idx0, idx1, idx_f, idx_cf

    len_dataset = 0
    for _ in dataset:
        len_dataset += 1
    dataset = dataset.shuffle(len_dataset)

    ite_scores = tf.Variable(tf.zeros((len_dataset, 1), dtype=tf.double))
    ate_scores = tf.Variable(tf.zeros((len_dataset, 2), dtype=tf.double))
    pehe_scores = tf.Variable(tf.zeros((len_dataset, 1), dtype=tf.double))
    y_error_val = tf.Variable(tf.zeros((len_dataset, 2), dtype=tf.double))

    for i, features in dataset.batch(params["batch_size"]).enumerate(0):
        """
        We need two arrays to select from other arrays. On the one hand
        we need to select the factual and counterfactual entries from
        the entries that are sorted by wheter or not the value of their
        intervention belongs to the first intervention or the zero'th
        intervention. After that we need to select from the y values
        that are sorted by being factual on wheter or not in the factual
        case we used the zero'th or first intervention value.
        """
        x_bin, x_cont, t, t_predict, y, y_predict, mu_0, mu_1 = features

        x = tf.concat([x_bin, x_cont], -1)
        # how do we define these in a dynamic way?
        # We need y and y_cf in some way. Or maybe not?
        t0 = t_predict[:, 0]
        t1 = t_predict[:, 1]
        y0 = y_predict[:, 0]
        y1 = y_predict[:, 1]

        idx0, idx1, idx_f, idx_cf = calculate_indices(t, t0, t1, y, y0, y1)
        true_ite = mu_1 - mu_0

        ypred0, ypred1 = model.do_intervention(x, t0, t1, n_samples)
        y_mean, y_std = scaling_data[0], scaling_data[1]
        ypred0, ypred1 = ypred0[:, -1:], ypred1[:, -1:]
        ypred0, ypred1 = ypred0 * y_std + y_mean, ypred1 * y_std + y_mean
        y = y[:, -1:]
        y = y * y_std + y_mean
        y_predict = y_predict[..., -1:]
        y_predict = y_predict * y_std + y_mean

        slice_indices = (i * features[0].shape[0],
                         (i + 1) * features[0].shape[0])
        ite_scores = ite_scores[slice_indices[0]:slice_indices[1]].\
            assign(rmse_ite(ypred1, ypred0, y, idx0, idx1))

        ate_scores = ate_scores[slice_indices[0]:slice_indices[1]].\
            assign(abs_ate(ypred1, ypred0, true_ite))

        pehe_scores = pehe_scores[slice_indices[0]:slice_indices[1]].\
            assign(pehe(ypred1, ypred0, mu_1, mu_0))

        y_error_val = y_error_val[slice_indices[0]:slice_indices[1]].\
            assign(y_errors(ypred1, ypred0, y_predict, idx_f, idx_cf))

    ite = tf.sqrt(tf.reduce_mean(ite_scores))
    ate = tf.abs(tf.reduce_mean(ate_scores[:, 0]) -
                 tf.reduce_mean(ate_scores[:, 1]))
    pehe = tf.sqrt(tf.reduce_mean(pehe_scores))
    y_rmse = tf.sqrt(tf.reduce_mean(y_error_val, axis=1))

    return ite, ate, pehe, y_rmse
