#import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

def get_log_prob(data, dist, mean=None, std=None, probs=None, name=None):

    if dist == 'N':
        assert mean is not None ,"No mean provided for distribution"
        if std is None:
            std = tf.ones_like(mean)
        distribution = tfd.Normal(loc=mean, scale=std)
    elif dist == 'B':
        assert probs is not None, "No probs provided for dsitribution"
        distribution = tfd.Bernoulli(probs=probs)
    else:
        raise NotImplementedError("Only Normal (N) and Bernoulli (B) have been implemeted")
        

    distribution = tfd.Independent(distribution, reinterpreted_batch_ndims=1, name=name)
    #return tf.reduce_mean(input_tensor=distribution.log_prob(data))
    return distribution.log_prob(data)

@tf.function
def get_analytical_KL_divergence(mean, std):
    KL = -0.5 * tf.reduce_sum(1 + tf.math.log(std ** 2) - mean ** 2 - std ** 2)
    return KL
