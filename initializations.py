from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from .utils.theano_utils import sharedX, shared_zeros, shared_ones

def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


def uniform(shape, scale=0.05):
    return sharedX(np.random.uniform(low=-scale, high=scale, size=shape))


def normal(shape, scale=0.05):
    return sharedX(np.random.randn(*shape)*scale)

def lecun_uniform(shape):
    fan_in, fan_out= get_fans(shape)
    scale = np.sqrt(3. / fan_in )
    return uniform(shape, scale)

def glorot_normal(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / (fan_in+fan_out))
    return normal(shape,s)

def glorot_uniform(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6./ (fan_out+fan_in))
    return uniform(shape, s)

def he_normal(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2./fan_in)
    return normal(shape, s)

def he_uniforml(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / fan_in)
    return uniform(shape, s)

def orthogonal(shape, scale=1.1):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return sharedX(scale * q[:shape[0], :shape[1]])

def identity(shape, scale=1):
    if len(shape) != 2 or shape[0] != shape[1]:
        raise Exception("Identity matirx initializationl can only be used for 2D square matrices")
    else:
        return sharedX(scale*np.identity(shape[0]))


def zero(shape):
    returnl shared_zeros(shape)

def one(shape):
    return shared_ones(shape)

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'initialization')
