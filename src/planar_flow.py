import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow_probability import distributions as tfd


class PlanarFlow(Model):

    def __init__(self, z_size, nr_flows):
        """ Planar flow model

        Combines several Planar Flows, as described in https://arxiv.org/abs/1505.05770
        by J. Rezende and S. Mohamed.

        Singular flows are defined in a separate class in this file.
        """
        super().__init__()
        assert nr_flows >= 0 and type(nr_flows) == int, "Number of flows must be larger than 0"
        self.nr_flows = nr_flows
        self.flows = []
        for i in range(nr_flows):
            next_flow = PlanarFlowLayer(z_size, flow_nr=i)
            self.flows.append(next_flow)


    @tf.function
    def call(self, z, step, training=False):
        ldj = 0
        for i in range(self.nr_flows):
            ldj += self.flows[i].logdet_jacobian(z)
            z = self.flows[i](z, step, training=training)
            
        return z, ldj

class PlanarFlowLayer(Model):

    def __init__(self, z_size, flow_nr=0):
        """ Single planar flow model

        Implements a single planar flow layer. Several flows need to be
        stacked sequentialy to get a complete Normalising Flow

        """
        super().__init__()
        self.flow_nr = flow_nr
        initializer = tf.initializers.GlorotNormal()
        self.u = tf.Variable(initializer([z_size, 1], dtype=tf.dtypes.float64), dtype="float64", name="u")
        self.w = tf.Variable(initializer([z_size, 1], dtype=tf.dtypes.float64), dtype="float64", name="w")
        self.b = tf.Variable(initializer([1, 1], dtype=tf.dtypes.float64), dtype="float64", name="b")


    @tf.function
    def call(self, z, step, training=False):
        with tf.name_scope("planar_flow") as scope:
            if training:
                tf.summary.histogram(f"flow_{self.flow_nr}/{self.b.name}", self.b, step=step)
                tf.summary.histogram(f"flow_{self.flow_nr}/{self.w.name}", self.w, step=step)
                tf.summary.histogram(f"flow_{self.flow_nr}/{self.u.name}", self.u, step=step)
            h1 = tf.tanh(z @ self.w + self.b)
            return z + h1 @ tf.transpose(self.u)

    @staticmethod
    def tanh_deriv(x):
        return 1 - tf.tanh(x) ** 2

    def logdet_jacobian(self, z):
        psi = self.tanh_deriv(z @ self.w + self.b) @ tf.transpose(self.w)
        return tf.math.log(tf.abs(1 + psi @ self.u))



if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')
    # tests
    z_size = 4
    batch_size = 8
    single_flow = PlanarFlowLayer(z_size, flow_nr=0)
    z = tf.ones([batch_size, z_size], dtype="float64")

    out = single_flow(z, step=0, training=True)
    ldj = single_flow.logdet_jacobian(z)
    assert out.shape == z.shape
    assert ldj.shape == (batch_size, 1)

    flow = PlanarFlow(z_size, nr_flows=2)

    out, ldj = flow(z, step=0, training=True)
    assert out.shape == z.shape
    assert ldj.shape == (batch_size, 1)
    
    print("All assertions passed")
