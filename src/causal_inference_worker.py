import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
# from tensorflow_probability import distributions as tfd


class CIWorker(Model):
    """ Base class for the causal inference models. """

    def __init__(self, params, category_sizes):
        Model.__init__(self)
        self.x_cat_size = params["x_cat_size"]
        self.x_cont_size = params["x_cont_size"]
        self.z_size = params["z_size"]
        self.debug = params["debug"]
        self.dataset_distributions = params["dataset_distributions"]
        self.category_sizes = category_sizes
        self.cumulative_sizes = np.cumsum(category_sizes)

    def elbo(self, features, output, step, params):
        raise NotImplementedError("elbo must be implemented in child class")

    def call(self, features, step, training=False):
        raise NotImplementedError("Forward pass must be implemented in child class")

    def grad(self, features, step, params):
        with tf.GradientTape() as tape:
            output = self(features, step, training=True)
            loss = self.elbo(features, *output, step, params)
        if self.debug:
            print(f"Forward pass complete, step: {step}")
        return loss, tape.gradient(loss, self.trainable_variables)
