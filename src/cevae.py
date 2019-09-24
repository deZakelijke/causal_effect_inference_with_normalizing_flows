import numpy as np
import torch
from torch import nn
import torchvision

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

class CEVAE(nn.module):
    """ CEVAE model with fc nets between random variables.
    """

    def __init__(self):
        super.__init__()
