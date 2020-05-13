import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from cevae import CEVAE
from cenf import CENF
from causal_real_nvp import CausalRealNVP


class CIWorker(Model):
    """ Base class for the causal inference models. """

    def __init__(self, params, category_sizes):
        super().__init__()
        self.x_cat_size = params["x_cat_size"]
        self.x_cont_size = params["x_cont_size"]
        self.z_size = params["z_size"]
        self.debug = params["debug"]
        self.dataset_distributions = params["dataset_distributions"]
        self.category_sizes = category_sizes
        self.model_type = params["model"]
        # self.cumulative_sizes = np.cumsum(category_sizes)

        if self.model_type == "cevae":
            self.model = CEVAE(params, category_sizes,
            hidden_size=params["hidden_size"],
            debug=self.debug)
        elif self.model_type == "cenf":
            self.model = CENF(params, category_sizes,
            hidden_size=params["hidden_size"],
            debug=self.debug)
        elif self.model_type == "crnvp":
            intervention_dims = 1
            if params['dataset'] == "SHAPES":
                dims_x = params["x_cont_size"]
                dims_y = params["x_cont_size"]
                architecture_type = "ResNet"
            else:
                dims_x = params["x_cat_size"] * category_sizes +\
                         params["x_cont_size"]
                dims_y = 1
                architecture_type = "FC_net"
            self.model = CausalRealNVP(dims_x, dims_y, intervention_dims,
                                       "CRNVP", params['hidden_size'],
                                       params["n_flows"],
                                       architecture_type=architecture_type,
                                       debug=self.debug)

    # @tf.function
    def call(self, features, step, training=False):
        if self.model_type == "crnvp":
            x = tf.concat([features[0], features[1]], axis=-1)
            t = features[2]
            y = features[3]
            return self.model(x, t, y, step, training)
        else:
            return self.model(features, step, training)

    def grad(self, features, step, params):
        with tf.GradientTape() as tape:
            output = self(features, step, training=True)
            loss = self.model.loss(features, *output, step, params)
        if self.debug:
            print(f"Forward pass complete, step: {step}")
        return loss, tape.gradient(loss, self.trainable_variables)

    def do_intervention(self, x, nr_samples):
        return self.model.do_intervention(x, nr_samples)
