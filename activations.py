from __future__ import absolute_import
import mxnet as mx
import theano.tensor as T


def softmax(x):
    return T.nnet.Softmax(x.reshape((-1,x.shape[-1]))).reshape(x.shape)


def time_distribute_softmax(x):
    import warnings
    warnings.warn("Deprecated . Use softmax", DeprecationWarning)
    return softmax(x)


def softplux(x):
    return T.nnet.softplut(x)

def prelu(x):
    if(x>0):
        return (x+abs(x))/2.0
    else:
        return -0.2

def relu(x):
    return (x+abs(x))/2.0


def tanh(x):
    return T.tanh(x)


def sigmoid(x):
    return T.nnet.sigmoid(x)

def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)

def linear(x):
    return x

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'activation functions')
