import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow import math
from tensorflow_probability import distributions as tfd

class RadialFlow(Model):
    """ Radial flow model

    Combines several Radial Flows, as described in
    https://arxiv.org/abs/1505.05770 by J. Rezende and S. Mohamed.

    Singular flows are defined in a separate class in this file.
    """

    def __init(self, z_dims, n_flows):
        """
        Parameters
        ----------
        z_dims : int
            Number of latent dimensions of the flow.
        n_flows : int
            Number of planar flows in the model.
        """

        super().__init__()
        assert nr_flows >= 0 and type(nr_flows) == int,\
            "Number of flows must be larger than 0"
        self.nr_flows = nr_flows
        if type(z_dims) == tuple:
            self.first_layer = layers.Flatten()
        else:
            self.first_layer = tf.identity
        z_dims = tf.cast(tf.reduce_prod(z_dims), tf.int32).numpy()

        self.flows = []
        for i in range(nr_flows):
            next_flow = RadialFlowLayer(z_dims, flow_nr=i)
            self.flows.append(next_flow)

    @tf.function
    def call(self, z, step, training=False):
        in_shape = z.shape
        z = self.first_layer(z)
        ldj = 0
        for i in range(self.nr_flows):
            ldj += self.flows[i].logdet_jacobian(z)
            z = self.flows[i](z, step, training=training)
        z = tf.reshape(z, in_shape)
        return z, ldj


class RadialFlowLayer(Model):
    """ Single radial flow model

    Implements a single radial flow layer. Several flows need to be
    stacked sequentialy to get a complete Normalising Flow
    """
    def __init__(self, z_dims, flow_nr=0):
        """
        Parameters
        ----------
        z_dims : int
            Number of latent dimensions of the flow.
        flow_nr : int
            Index of the flow in the larger set of flows. Used for indexing
            in Tensorboard.
        """

        super().__init__()
        self.flow_nr = flow_nr
        initializer = tf.initializers.GlorotNormal()
        self.z_0 = tf.Variable(initializer([z_dims, 1], dtype=tf.dtypes.float64),
                               dtype="float64", name="z_0")
        self.a = tf.Variable(initializer([1, 1], dtype=tf.dtypes.float64),
                             dtype="float64", name="a")
        self.b = tf.Variable(initializer([1, 1], dtype=tf.dtypes.float64),
                             dtype="float64", name="b")

    @tf.function
    def call(self, z, step, training=False):
        uw = tf.transpose(self.w) @ self.u
        norm_w = tf.transpose(self.w) @ self.w
        u = self.u + (-1 + math.softplus(uw) - uw) * self.w / norm_w

        with tf.name_scope("planar_flow") as scope:
            if training and step is not None:
                tf.summary.histogram(f"flow_{self.flow_nr}/{self.b.name}",
                                     self.b, step=step)
                tf.summary.histogram(f"flow_{self.flow_nr}/{self.a.name}",
                                     self.a, step=step)
                tf.summary.histogram(f"flow_{self.flow_nr}/{self.z_0.name}",
                                     self.z_0, step=step)
            diff = z - self.z_0
            r = tf.norm(diff)
            h = 1 / (self.a + r)
            return z + self.b * h * diff

    def logdet_jacobian(self, z):
        
        return 
