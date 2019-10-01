import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.activations import softplus
from tensorflow_probability import distributions as tfd
from fc_net import FC_net


"""
Old structure:
    for i in get_train_valid_test
        split into  train, val and test

        reorder features to have binary first

        Make the values of y have zero mean and unit variance during training
        
        Start session:
            Set random seeds
            x_ph = Make placeholders for binary and continuous inputs and concat them
            set activation to relu
            
            Define Decoder:
                p(z) 
                z = Normal(0,1)

                p(x|z)
                hx = fc_net(z)
                logits = fc_net(hx)
                x1 = Bernoulli(logits)
                mu, sigma = fc_net(hx)
                x2 = Normal(mu, sigma)

                p(t|z)
                logits = fc_net(z)
                t = Bernoulli(logits)

                p(y|t,z)
                mu2_t0 = fc_net(z)
                mu2_t1 = fc_net(z)
                y = Normal(t * mu2_t1 + (1 - t) * mu2_t0, 1)

            Define Encoder:
                q(t|x)
                logits_t = fc_net(x_ph)
                qt = Bernoulli(logits_t)

                q(y|x,t)
                hqy = fc_net(x_ph)
                mu_qy_t0 = fc_net(hqy)
                mu_qy_t1 = fc_net(hqy)
                qy = Normal(qt * mu_qy_t1 + (1 - qt) * mu_qy_t0, 1)

                q(z|x,t,y)
                inpt2 = concat(x_ph, qy)
                hqz = fc_net(inpt2)
                muq_t0, sigmaq_t0 = fc_net(hqz)
                muq_t1, sigmaq_t1 = fc_net(hqz)
                qz = Normal(qt * muq_tq + (1 - qt) * muq_t0, qt * sigmaq_t1 + (1 - qt) * sigmaq_t0)

            Something with a dict for the Edward moduel

            Make a deterministic output for early stopping
            inference = KL_divergence_optimizer(z: qz, dict_with_data) (This does the VAE optimizations)
            inference.init(optim=Adam)

            something else with disctionaries for evaluation

            for epoch in nr_epochs
                reset average loss
                reset timer and progressbar
                for j in interations per epoch
                    batch = sample random batch
                    loss = inference.update(batch)
                    average loss += loss
                average loss /= iterations per eopch * 100

                if iteration to print
                    print logs of data

            Save session?
            Print some scores

"""


def make_cevae(x_bin_size, x_cont_size, z_size, hidden_size=512):
    """ CEVAE model with fc nets between random variables.
    https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py
    """

        # Encoder part
        #qt       = tfd.Bernoulli(logits=FC_net(1, hidden_size), dtype=tf.float32)
        qt_logits = FC_net(1, hidden_size)
        hqy      = FC_net(hidden_size, hidden_size)
        mu_qy_t0 = FC_net(1, hidden_size)
        mu_qy_t1 = FC_net(1, hidden_size)
        #qy       = tfd.Normal(loc=qt * mu_qy_t1 + (1. - qt) * mu_qy_t0, scale=tf.ones_like(mu_qy_t0))

        hqz   = FC_net(z_size, hidden_size)
        qz_t0 = FC_net(z_size * 2, hidden_size)
        qz_t1 = FC_net(z_size * 2, hidden_size)
        #qz    = tfd.Normal(loc=self.qt * qz_t1[:, :z_size] + (1. - self.qt) * qz_t0[:, :z_size],
                scale=self.qt * softplus(qz_t1[:, z_size:]) + (1. - self.qt) * softplus(qz_t0[:, z_size:]))

        # Decoder part
        #pz = tfd.Normal(loc=tf.zeros([z_size]), scale=tf.ones([z_size]))
        hx      = FC_net(x_bin_size + x_cont_size, hidden_size)
        #x1      = tfd.Bernoulli(logits=FC_net(x_bin_size, hidden_size), dtype=tf.float32)
        x_cont_logits = FC_net(x_cont_size, hidden_size)
        hx_c    = FC_net(x_cont_size * 2, hidden_size)
        #x2      = tfd.Normal(loc=hx_c[:, :x_cont_size], scale=softplus(hx_c[:, x_cont_size:]))
        x_bin_logits = FC_net(x_bin_size, hidden_size)
        #t       = tdf.Bernoulli(logits=FC_net(1, hidden_size), dtype=tf.float32)
        t_logtis = FC_net(1, hidden_size)
        mu_y_t0 = FC_net(1, hidden_size)
        mu_y_t1 = FC_net(1, hidden_size)
        #y       = tfd.Normal(loc=t * mu_y_t1 + (1. - t) * mu_y_t0, scale=tf.ones_like(mu_y_t0))


    def encode(self, x, y, t):
        qt = tfd.Bernoulli(qt_logits(x))
        mu0 = mu_qy_t0(hqy(qt))
        mu1 = mu_qy_t1(hqy(qt))
        qy = tdf.Normal(qt * mu1 + (1. - qt) * mu0, scale-tf.ones_like(mu0))




    return encode, decode




