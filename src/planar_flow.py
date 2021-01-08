import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow import math
from tensorflow_probability import distributions as tfd

from cenf import CENF


class PlanarFlow(CENF):
    """ Planar flow model

    Combines several Planar Flows, as described in
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
        name_tag="Planar flow",
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
            next_flow = PlanarFlowLayer(z_dims, flow_nr=i)
            self.z_flows.append(next_flow)


class PlanarFlowLayer(Model):
    """ Single planar flow model

    Implements a single planar flow layer. Several flows need to be
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
        self.u = tf.Variable(initializer([z_dims, 1], dtype=tf.dtypes.float64),
                             dtype="float64", name="u")
        self.w = tf.Variable(initializer([z_dims, 1], dtype=tf.dtypes.float64),
                             dtype="float64", name="w")
        self.b = tf.Variable(initializer([1, 1], dtype=tf.dtypes.float64),
                             dtype="float64", name="b")

    @tf.function
    def call(self, z, step, training=False):
        uw = tf.transpose(self.w) @ self.u
        norm_w = tf.transpose(self.w) @ self.w
        u = self.u + (-1 + math.softplus(uw) - uw) * self.w / norm_w

        with tf.name_scope("planar_flow") as scope:
            # if training and step is not None and step % 50 == 0:
            #     tf.summary.histogram(f"flow_{self.flow_nr}/{self.b.name}",
            #                          self.b, step=step)
            #     tf.summary.histogram(f"flow_{self.flow_nr}/{self.w.name}",
            #                          self.w, step=step)
            #     tf.summary.histogram(f"flow_{self.flow_nr}/{self.u.name}",
            #                          self.u, step=step)
            h1 = tf.tanh(z @ self.w + self.b)
            return z + h1 @ tf.transpose(u)

    @staticmethod
    def tanh_deriv(x):
        return 1 - tf.tanh(x) ** 2

    def logdet_jacobian(self, z):
        uw = tf.transpose(self.w) @ self.u
        norm_w = tf.transpose(self.w) @ self.w
        u = self.u + (-1 + math.softplus(uw) - uw) * self.w / norm_w

        psi = self.tanh_deriv(z @ self.w + self.b) @ tf.transpose(self.w)
        return math.log(tf.abs(1 + psi @ u))


def test_flow():
    z_dims = 4
    batch_size = 8
    single_flow = PlanarFlowLayer(z_dims, flow_nr=0)
    z = tf.ones([batch_size, z_dims], dtype="float64")

    out = single_flow(z, step=0, training=True)
    ldj = single_flow.logdet_jacobian(z)
    assert out.shape == z.shape
    assert ldj.shape == (batch_size, 1)

    flow = PlanarFlow(z_dims, n_flows=4)

    out, ldj = flow(z, step=0, training=True)
    assert out.shape == z.shape
    assert ldj.shape == (batch_size, 1)

    print("All assertions passed, test successful")


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')
    test_flow()
