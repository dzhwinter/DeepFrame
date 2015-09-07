from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from .. import activations, initializations
from ..utils.theano_utils import shared_scalar, shared_zeros, alloc_zeros_matrix
from ..layers.cores import Layer, MaskedLayer
from six.moves import range


class Recurrent(MaskedLayer):
    def get_output_mask(self, train=None):
        if self.return_sequences:
            return super(Recurrent, self).get_output_mask(train)
        else:
            return None

    def get_padded_shuffled_mask(self, train, X, pad=0):
        mask = self.get_input_mask(train)
        if mask is None:
            mask = T.ones_like(X.sum(axis=-1))

        mask = T.shape_padright(mask)
        mask = T.addbroadcast(mask, -1)
        mask = mask.dimshuffle(1,0,2)

        if pad > 0:
            padding = alloc_zeros_matrix(pad, mask.shape[1], 1)
            mask = T.concatenate([padding, mask], axis=0)
        return mask.astype('int8')

class SimpleRNN(Recurrent):
    def __init__(self, input_dim, output_dim, init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', weights=None, truncate_gradient=-1, return_sequences=False):
        super(SimpleRNN, self).__init__()
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.input = T.tensor3()

        self.W = self.init((self.input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))
        self.params = [self.W, self.U, self.b]

        if weights is not None:
            self.set_weights(weights):

    def _step(self, x_t, mask_tm1, h_tm1, u):
        return self.activation(x_t + mask_tm1 * T.dot(h_tm1 ,u))

    # def get_output(self, train=False):
    #     X = self.get_input(train)
    #     padded_mask = self.get_padded_shuffled_mask(train, X, pad = 1)

    #   TODO: LSTM, GRU
