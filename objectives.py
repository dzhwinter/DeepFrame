from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
from six.moves import range


if theano.config.floatX == 'float64':
    epsilon = 1.0e-9
else:
    epsilon = 1.0e-7


def mean_squared_error(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean(axis=-1)


def mean_absolute_error(y_true, y_pred):
    return T.abs_(y_pred - y_true).mean(axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    return T.abs_((y_true - y_pred) / T.clip(T.abs_(y_true), epsilon, np.inf)).mean(axis=-1) * 100

def mean_sqaured_logarithmic_error(y_true, y_pred):
    return T.sqr(T.log(T.clip(y_pred, epsilon, np.inf) + 1.) - T.log(T.clip(t_true, epsilon, np.inf) + 1.)).mean(axis=-1)

def square_hinge(y_true, y_pred):
    return T.sqr(T.maximum(1. - y_true * y_pred,  0.)).mean(axis=-1)

def hinge(y_true, y_pred):
    return T.maximum(1. - y_true * y_pred, 0.).mean(axis=-1)


def categorical_crossentropy(y_true, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0-epsilon)
    y_pred /= y_pred.sum(axis=-1, keepdims=True)
    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    return cce


def binary_crossentropy(y_true, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    bce = T.nnet.binary_crossentropy(y_pred, y_true).mean(axis=-1)
    return bce

def posison_loss(y_true, y_pred):
    return T.mean(y_pred - y_true * T.log(y_pred + epsilon), axis=-1)

mse = MSE = mean_sqaured_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_sqaured_logarithmic_error

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')
