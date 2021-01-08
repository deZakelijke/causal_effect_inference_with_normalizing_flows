import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow import math
from tensorflow_probability import distributions as tfd

from cenf import CENF


class RadialFlow(CENF):
    """ Radial flow model

    Combines several Radial Flows, as described in
    https://arxiv.org/abs/1505.05770 by J. Rezende and S. Mohamed.

    Singular flows are defined in a separate class in this file.
    """

    def __init__(
        self,
        x_dims=30,
        x_cat_dims=10,
        x_cont_dims=10,
        t_dims=2,
        t_type='Categorical',
        y_dims=1,
        y_type='Normal',
        z_dims=32,
        category_sizes=2,
        n_flows=2,
        name_tag="Radial flow",
        feature_maps=256,
        architecture_type="FC_net",
        log_steps=10,
        flow_type_variational="PlanarFlow",
        debug=False,
        **_
    ):

        """
        Parameters
        ----------
        z_dims : int
            Number of latent dimensions of the flow.
        n_flows : int
            Number of planar flows in the model.
        """
        super().__init__(
            x_dims=x_dims,
            x_cat_dims=x_cat_dims,
            x_cont_dims=x_cont_dims,
            t_dims=t_dims,
            t_type=t_type,
            y_dims=y_dims,
            y_type=y_type,
            z_dims=z_dims,
            category_sizes=category_sizes,
            name_tag=name_tag,
            feature_maps=feature_maps,
            architecture_type=architecture_type,
            log_steps=log_steps,
            debug=debug
        )

        self.name_tag = name_tag
        assert n_flows >= 0 and type(n_flows) == int,\
            "Number of flows must be larger than 0"
        self.n_flows = n_flows
        if type(z_dims) == tuple:
            self.flow_start_layer = layers.Flatten()
        else:
            self.flow_start_layer = tf.identity
        z_dims = tf.cast(tf.reduce_prod(z_dims), tf.int32).numpy()

        self.z_flows = []
        for i in range(n_flows):
            next_flow = RadialFlowLayer(z_dims, flow_nr=i)
            self.z_flows.append(next_flow)


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
        self.z_0 = tf.Variable(initializer([1, z_dims],
                                           dtype=tf.dtypes.float64),
                               dtype="float64", name="z_0")
        self.a = tf.Variable(initializer([1, 1], dtype=tf.dtypes.float64),
                             dtype="float64", name="a")
        self.b = tf.Variable(initializer([1, 1], dtype=tf.dtypes.float64),
                             dtype="float64", name="b")

    @tf.function
    def call(self, z, step, training=False):
        a = math.softplus(self.a)
        b = -a + math.softplus(self.b)
        with tf.name_scope("radial_flow") as scope:
            # if training and step is not None and step % 50 == 0:
            #     tf.summary.histogram(f"flow_{self.flow_nr}/{self.b.name}",
            #                          self.b, step=step)
            #     tf.summary.histogram(f"flow_{self.flow_nr}/{self.a.name}",
            #                          self.a, step=step)
            #     tf.summary.histogram(f"flow_{self.flow_nr}/{self.z_0.name}",
            #                          self.z_0, step=step)
            diff = z - self.z_0
            r = tf.norm(diff, axis=1, keepdims=True)
            h = 1 / (a + r)
            return z + b * h * diff

    def logdet_jacobian(self, z):
        a = math.softplus(self.a)
        b = -a + math.softplus(self.b)
        d = z.shape[1]
        r = tf.norm(z - self.z_0, axis=1, keepdims=True)
        h = 1 / (a + r)
        h_prime = -1 / math.square(a + r)
        b_h = b * h
        ldj = (d - 1) * math.log(1 + b_h) + math.log(1 + b_h + b * h_prime * r)
        return ldj


def test_flow():
    z_dims = 4
    batch_size = 8
    single_flow = RadialFlowLayer(z_dims, flow_nr=0)
    z = tf.ones([batch_size, z_dims], dtype="float64")

    out = single_flow(z, step=0, training=True)
    ldj = single_flow.logdet_jacobian(z)
    assert out.shape == z.shape
    assert ldj.shape == (batch_size, 1), f"ldj shape: {ldj.shape}"

    flow = RadialFlow(z_dims, n_flows=4)

    out, ldj = flow(z, step=0, training=True)
    assert out.shape == z.shape
    assert ldj.shape == (batch_size, 1)

    print("All assertions passed, test successful")


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')
    test_flow()
