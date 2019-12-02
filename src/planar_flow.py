import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow_probability import distributions as tfd


class PlanarFlow(Model):

    def __init__(self, z_size):
        """ Planar flow model

        Implements a sngle planar flow layer. Several flows need to be
        stacked sequentialy to get a complete Normalising Flow

        """
        super().__init__()
        initializer = tf.initializers.GlorotNormal()
        self.u = tf.Variable(initializer([z_size, 1], dtype=tf.dtypes.float64), dtype="float64", name="planar_flow/u")
        self.w = tf.Variable(initializer([z_size, 1], dtype=tf.dtypes.float64), dtype="float64", name="planar_flow_w")
        self.b = tf.Variable([0], dtype="float64", name="planar_flow/b")


    @tf.function
    def call(self, z):
        #h1 = tf.tanh(tf.transpose(self.w) @ z + self.b)
        #return z + self.u * h1
        h1 = tf.tanh(z @ self.w + self.b)
        return z + h1 @ tf.transpose(self.u)

    @staticmethod
    def tanh_deriv(x):
        return 1 - tf.tanh(x) ** 2

    def logdet_jacobian(self, z):
        #psi = self.tanh_deriv(tf.transpose(self.w) @ z + self.b) * self.w
        #return tf.abs(1 + tf.transpose(u) @ psi)
        psi = self.tanh_deriv(z @ self.w + self.b) @ tf.transpose(self.w)
        return tf.math.log(tf.abs(1 + psi @ self.u))



if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')
    # tests
    z_size = 4
    batch_size = 8
    flow = PlanarFlow(z_size)
    z = tf.ones([batch_size, z_size], dtype="float64")

    out = flow(z)
    ldj = flow.logdet_jacobian(z)
    assert out.shape == z.shape
    assert ldj.shape == (batch_size, 1)
    print(out)
    print(ldj)
