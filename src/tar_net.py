import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError

from fc_net import FC_net
from res_net import ResNet


class TARNET(Model):
    """ Implementation of Treatment-Agnostic Representation Network

    This is basically a multi-headed network, where we have a head for each
    possible value of the intervention. That is quite straightforward, but we
    have the same problem as in the CEVAE when the intervention is continuous.
    How do we solve that?
    """

    def __init__(
        self,
        x_dims=30,
        t_dims=2,
        y_dims=1,
        y_type='Normal',
        category_sizes=2,
        name_tag="no_name",
        feature_maps=256,
        architecture_type="FC_net",
        log_steps=10,
        debug=False,
        **_
    ):

        """
        One net from x to some feature vector. Concatenate t and then predict y
        """

        super().__init__(name=name_tag)
        self.debug = debug
        self.category_sizes = category_sizes
        self.t_dims = t_dims
        self.y_dims = y_dims
        self.log_steps = log_steps
        self.architecture_type = architecture_type

        self.annealing_factor = 1.0  # dummy variable

        if y_type == "Categorical":
            self.y_loss = CategoricalCrossentropy()
            self.y_activation = nn.softmax
        else:
            self.y_loss = MeanSquaredError()
            self.y_activation = lambda x: x

        if architecture_type == "ResNet":
            intermediate_dims = feature_maps
            self.x_network = ResNet(in_dims=x_dims, out_dims=x_dims,
                                    name_tag="x_net", n_layers=3,
                                    feature_maps=feature_maps,
                                    squeeze=True,
                                    squeeze_dims=intermediate_dims,
                                    debug=debug)
        else:
            intermediate_dims = feature_maps // 2
            self.x_network = FC_net(in_dims=x_dims, out_dims=intermediate_dims,
                                    name_tag="x_net", n_layers=3,
                                    feature_maps=feature_maps, debug=debug)
        self.prediction_network = FC_net(in_dims=intermediate_dims + t_dims,
                                         out_dims=y_dims, name_tag="y_net",
                                         n_layers=3, feature_maps=feature_maps,
                                         debug=debug)

    @tf.function
    def call(self, x, t, y, step, training=False):
        """
        Pass x through network
        Pass t and new representation of x through network
        Ignore y at this point
        """
        intermediate_vec = self.x_network(x, step, training=training)
        xt = tf.concat([intermediate_vec, t], -1)
        y_pred = self.y_activation(self.prediction_network(xt,
                                                           step,
                                                           training=training))
        return (y_pred, )

    @tf.function
    def loss(self, features, y_pred, step):
        _, _, _, _, y, *_ = features
        distortion_y = self.y_loss(y, y_pred)
        loss = tf.reduce_mean(distortion_y)

        if step is not None and step % (self.log_steps * 5) == 0:
            l_step = step // (self.log_steps * 5)
            tf.summary.scalar("partial_loss/distortion_y",
                              loss, step=l_step)
        return loss

    def do_intervention(self, x, t0, t1, n_samples):
        intermediate_vec = self.x_network(x, None, training=False)
        xt0 = tf.concat([intermediate_vec, t0], -1)
        xt1 = tf.concat([intermediate_vec, t1], -1)
        y0 = self.prediction_network(xt0, None, training=False)
        y1 = self.prediction_network(xt1, None, training=False)

        return self.y_activation(y0), self.y_activation(y1)
