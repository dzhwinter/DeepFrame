from __future__ import absolute_import
import numpy as np
from .cores import MaskedLayer

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class GaussainNoise(MaskedLayer):
    def __init__(self, sigma):
        super(GaussainNoise, self).__init__()
        self.sigma = sigma
        self.srng = RandomStreams(seed=np.random.randint(10e6))

    def get_output(self, train=False):
        x = self.get_input(train)
        if not train or self.sigma == 0:
            return x
        else:
            return X + self.srng.normal(size=X.shape, avg=0.0, std=self.sigma,
                                        dtype=theano.config.floatX)

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "sigma": self.sigma
        }

class GaussainDropout(MaskedLayer):
    def __init__(self, p):
        super(GaussainDropout, self).__init__()
        self.p = p
        self.srng = RandomStreams(seed=np.random.randint(10e6))

    def get_output(self, train):
        X = self.get_input(train)
        if train:
            X *= self.srng.normal(size=X.shape, avg=1.0, std=T.sqrt(self.p/(1.0-self.p)), dtype=theano.config.floatX)
        return X

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "p": self.p
        }
                                
