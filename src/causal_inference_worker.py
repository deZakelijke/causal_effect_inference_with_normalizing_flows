import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from cevae import CEVAE
from cenf import CENF


class CIWorker(Model):
    """ Base class for the causal inference models. """

    def __init__(self, params, category_sizes, debug=False):
        super().__init__()
        self.x_cat_size = params["x_cat_size"]
        self.x_cont_size = params["x_cont_size"]
        self.z_size = params["z_size"]
        self.debug = params["debug"]
        self.dataset_distributions = params["dataset_distributions"]
        self.category_sizes = category_sizes
        # self.cumulative_sizes = np.cumsum(category_sizes)

        if params['model'] == "cevae":
            self.model = CEVAE(params, category_sizes, debug=debug)
        elif params['model'] == "cenf":
            self.model = CENF(params, category_sizes, debug=debug)

    def call(self, features, step, training=False):
        return self.model(features, step, training)

    def grad(self, features, step, params):
        with tf.GradientTape() as tape:
            output = self.model(features, step, training=True)
            loss = self.mode.loss(features, *output, step, params)
        if self.debug:
            print(f"Forward pass complete, step: {step}")
        return loss, tape.gradient(loss, self.trainable_variables)

    def do_intervention(self, x, nr_samples):
        raise NotImplementedError
