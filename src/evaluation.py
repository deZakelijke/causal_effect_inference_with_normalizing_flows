import numpy as np
import tensorflow as tf


#class Evaluator(object):
def calc_stats(model, dataset, params):
    """ Function for evaluating a model according to our metrics

   
    Function that calls all individual metrics that we wat to computes 
        and returns a tuple of all metrics.

        Args: 
            ypred1, ypred0: output of get_y0_y1, rescaled with original std and mean of y
    """

    def rmse_ite(ypred1, ypred0):
        pred_ite = tf.zeros_like(true_ite)
        idx1, idx0 = tf.where(t == 1), tf.where(t == 0)
        ite1, ite0 = y[idx1] - ypred0[idx1], ypred1[idx0] - y[idx0]
        pred_ite[idx1] = ite1
        pred_ite[idx0] = ite0
        return tf.sqrt(tf.reduce_mean(tf.square(true_ite - pred_ite)))

    def abs_ate(ypred1, ypred0):
        return tf.abs(tf.reduce_mean(ypred1 - ypred0) - tf.redce_mean(true_ite))

    def pehe(ypred1, ypred0):
        return tf.sqrt(tf.reduce_mean(tf.square((mu_1 - self.mu_0) - (ypred1 - ypred0))))

    def y_errors(y0, y1):
        ypred = (1 - t) * y0 + t * y1
        ypred_cf = t * y0 + (1 - t) * y1
        return y_errors_pcf(ypred, ypred_cf)

    def y_errors_pcf(ypred, ypred_cf):
        rmse_factual = tf.sqrt(tf.reduce_mean(tf.square(ypred - y)))
        rmse_cfactual = tf.sqrt(tf.reduce_mean(tf.square(ypred_cf - y_cf)))
        return rmse_factual, rmse_cfactual


    avg_ite = 0
    avg_ate = 0
    avg_pehe = 0

    for features in dataset.batch(params["batch_size"]):
        x_bin, x_cont, t, y, y_cf, mu_0, mu_1 = features
        true_ite = mu_1 - mu_0
 
   # def calc_stats(ypred1, ypred0):
   #     ite = self.rmse_ite(ypred1, ypred0)
   #     ate = self.abs_ate(ypred1, ypred0)
   #     pehe = self.pehe(ypred1, ypred0)
   #     return ite, ate, pehe


   # return calc_stats
