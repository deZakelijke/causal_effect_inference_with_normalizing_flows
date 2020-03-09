#import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

def get_log_prob(data, dist, mean=None, std=None, probs=None, name=None):
    """ Returns the log likelihood of a batch of samples for a given family of
        distributions and its parameters.

    """
    if dist == 'N':
        assert mean is not None ,"No mean provided for distribution"
        if std is None:
            std = tf.ones_like(mean)
        distribution = tfd.Normal(loc=mean, scale=std)
    elif dist == 'B':
        assert probs is not None, "No probs provided for dsitribution"
        distribution = tfd.Bernoulli(probs=probs)
    elif dist == 'M':
        assert probs is not None, "No probs provided for this distribution"
        distribution = tfd.OneHotCategorical(probs=probs, dtype=tf.float64)
    else:
        raise NotImplementedError("Only Normal (N), Bernoulli (B) and Categorical (M) have been implemeted")
        
    distribution = tfd.Independent(distribution, reinterpreted_batch_ndims=1, name=name)
    return distribution.log_prob(data)


@tf.function
def get_analytical_KL_divergence(mean, std):
    """ Return the analytical KL-divergence between a diagonal Gaussian defined
        by mean and std, and an isotropic unit Gaussian.
    """
    KL = -0.5 * tf.reduce_sum(1 + tf.math.log(std ** 2) - mean ** 2 - std ** 2, axis=1)
    return KL
