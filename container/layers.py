from __future__ import absolute_import

import theano
import theano.tensor as T
import numpy as np
from .. import activations, initializations, regularizers, constraints
from ..utils.theano_utils import shared_ones, shared_zeros, floatX
from ..utils.generic_utils import make_tuple
from ..regularizers import ActivityRegularizer, Regularizer

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from six.moves import zip


class Layer(object):
    #TODO: move to mshadow, add fast convolution
    '''
    core Layer attribute:
    -params
    -updates

    -nb_input
    -np_output
    :Maybe
    -constraints
    '''
    def __init__(self):
        self.params = []

    def init_updates(self):
        self.updates = []

    def set_previous(self, layer, connection_map={}):
        assert self.nb_input == layer.np_output == 1, "Cannot connect layers: input count and output count should be 1"
        if not self.supports_masked_input() and layer.get_output_mask() is not None:
            raise Exception(
                "Cannot connect non-masking layer to layer with masked output")

    @property
    def nb_input(self):
        return 1

    @property
    def nb_output(self, train=False):
        return self.get_input(train)

    def get_output(self, train=False):
        return self.get_input(train)

    def get_input(self, train=False):
        if hasattr(self, 'previous'):
            return self.previous.get_output(train=train)
        else:
            return self.input

    # this support is tested
    def supports_masked_input(self):
        '''
        attach something to its previous layer
        '''
        return False

    def get_output_mask(self, train=None):
        return None

    def set_weights(self, weights):
        for p, w in zip(self.params, weights):
            if p.eval().shape != w.shape:
                raise Exception(
                    "Layer shape %s not compatible with weight shape %s." %
                    (p.eval().shape, w.shape))
            p.set_value(floatX(w))

    def get_weights(self):
        weight = []
        for p in self.params:
            weight.append(p.get_value())
        return weight

    def get_config(self):
        return {"name": self.__class__.__name__}

    # def get_config(self):
    def get_params(self):
        consts = []
        updates = []
        if hasattr(self, 'regularizers'):
            regularizers = self.regularizers
        else:
            regularizers = []

        if hasattr(self, 'constraints') and len(self.constraints) == len(self.params):
            for c in self.constraints:
                if c:
                    consts.append(c)
                else:
                    consts.append(constraints.identity())
        elif hasattr(self, 'constraints') and self.constraint:
            consts += [self.constraint for _ in range(len(self.params))]
        else:
            consts += [constraints.identity() for _ in range(len(self.params))]

        if hasattr(self, 'updates') and self.updates:
            updates += self.updates

        return self.params, regularizers, consts, updates

    def set_name(self, name):
        for i in range(len(self.params)):
            self.params[i].name = '%s_p%d' % (name, i)


class Merge(Layer):
    def __init__(self, layers, mode='sum'):
        '''
        Merge the output of a list of layers or containers into a single tensor
        mode : {'sum', 'concat'}
        '''
        if len(layers) < 2:
            raise Exception(
                "Please specify two or more input layers (or containers) to merge")
        self.mode = mode
        self.layers = layers
        self.params = []
        self.regularizers = []
        self.constraints = []
        self.updates = []
        for l in self.layers:
            params, regs, consts, updates = l.get_params()
            self.regularizers += regs
            self.updates += updates
            for p, c in zip(params, consts):
                if p not in self.params:
                    self.params.append(p)
                    self.constraints.append(c)

        def get_params(self):
            return self.params, self.reuglarizers, self.constraints, self.updates

        def get_output(self, train=False):
            if self.mode == 'sum':
                s = self.layers[0].get_output(train)
                for i in range(1, len(self.layers)):
                    s += self.layers[i].get_output(train)
                return s
            elif self.mode == 'concat':
                inputs = [self.layers[i].get_output(train)
                          for i in range(len(self.layers))]
                return T.concatenate(inputs, axis=-1)
            else:
                raise Exception('Unknown merge mode')

        def get_input(self, train=False):
            res = []
            for i in range(len(self.layers)):
                o = self.layers[i].get_input(train)
                if not type(o) == list:
                    o = [o]
                for output in o:
                    if output not in res:
                        res.append(output)
            return res

        @property
        def input(self):
            return self.get_input()

        def supports_masked_input(self):
            return False

        def get_output_mask(self, train=None):
            return None

        def get_weights(self):
            weights = []
            for l in self.layers:
                weights += l.get_weights()
            return weights

        def set_weights(self, weights):
            for i in range(len(self.layers)):
                nb_param = len(self.layers[i].params)
                self.layers[i].set_weights(weights[:nb_param])
                weights = weights[nb_param]

        def get_config(self):
            return {
                "name": self.__class__.__name__,
                "layers": [l.get_config() for l in self.layers],
                "mode": self.mode
            }


class MaskedLayer(Layer):
    def supports_masked_input(self):
        return True

    def get_input_mask(self, train=False):
        if hasattr(self, 'previous'):
            return self.previous.get_output_mask(train)
        else:
            return None

    def get_output_mask(self, train=False):
        return self.get_input_mask(train)


class Masking(Maskedlayer):
    def __init__(self, mask_value=0.):
        super(Masking, self).__init__()
        self.mask_value = mask_value
        self.input = T.tensor3()

    def get_output_mask(self, train=False):
        X = self.get_input(train)
        return T.any(T.ones_like(X) * (1. - T.eq(X, self.mask_value)), axis=-1)

    def get_output(self, train=False):
        X = self.get_input(train)
        return X * T.shape_padright(T.any((1. - T.eq(X, self.mask_value)),
                                          axis=-1))

    def get_config(self):
        return {"name": self.__class__.__name__, "mask_value": self.mask_value}


class Activation(Layer):
    def __init__(self, activation, target=0, beta=0.1):
        super(Activation, self).__init__()
        self.activation = activations.get(activation)
        self.target = target
        self.beta = beta

    def get_output(self, train=False):
        X = self.get_input(train)
        return self.activation(X)

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "activation": self.activation.__name__,
            "target": self.target,
            "beta": self.beta
        }

class Reshape(Layer):
    def __init__(self, *dims):
        super(Reshape, self).__init__()
        self.dims = dims

    def get_output(self, train=False):
        X = self.get_input(train)
        nshape = make_tuple(X.shape[0], *self.dims)
        return theano.tensor.reshape(X, nshape)


    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "dims": self.dims
        }

class Permute(Layer):
    def _init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def get_output(self, train):
        X = self.get_output(train)
        return X.dimshuffle((0, ) +  self.dims)

    def get_config(self):
        return{
            "name": self.__class__.__name__,
            "dims": self.dims
        }


class Flatten(Layer):
    def __init__(self):
        super(Flatten, self).__init__()

    def get_output(self, train=False):
        X = self.get_input(train)
        size = theano.tensor.prod(X.shape)
        nshape = (X.shape[0], size)
        return theano.tensor.reshape(X, nshape)



class RepeatVector(Layer):
    def __init__(self, n):
        super(RepeatVector, self).__init__()
        self.n = n

    def get_output(self, train=False):
        X = self.get_input(train)
        tensors = [X] * self.n
        stacked = theano.tensor.stack(*tensors)
        return stacked.dimshuffle((1,0,2))

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "n": self.n
        }

class Dense(Layer):
    def __init__(self, input_dim, output_dim, init='glorot_uniform', activation='linear',
                 weights=None, name=None,
                 W_regularizer=None, b_regularizer=None,
                 activity_regularizer=None, W_constraint=None, b_constraint=None):
        super(Dense, self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.matrix()
        self.W = self.init((self.input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.params = [self.W, self.b]

        self.regularizers = []
        self.W_regularizer = regularizers.get(W_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
            
        self.b_regularizer = regularizers.get(b_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_param(self.activity)
            self.regularizers.append(self.activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        if weights is not None:
            self.set_weights(weights)

        if name is not None:
            self.set_name(name)


    def set_name(self, name):
        self.W.name = '%s_W' %(name)
        self.b.name = '%s_b' %(name)

    def get_output(self, train=False):
        X = self.get_input(train)
        output = self.activation(T.dot(X, self.W) + self.b)
        return output

         return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "activation": self.activation.__name__,
                "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                "b_constraint": self.b_constraint.get_config() if self.b_constraint else None}

class ActivityRegularization(Layer):
    def __init__(self,l1=0., l2=0.):
        super(ActivityRegularization, self).__init__()
        self.l1 = l1
        self.l2 = l2

        activity_regularizer = ActivityRegularization(l1=l1, l2=l2)
        activity_regularizer.set_layer(self)
        self.regularizers = [activity_regularizer]

    def get_output(self, train=False):
        return self.get_input(train)

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "l1" : self.l1,
            "l2": self.l2
        }


class TimeDistributeDense(MaskedLayer):
    def __init__(self, input_dim, output_dim, init='glorot_uniform', activation='linear',
                 weights = None, W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None):

        super(TimeDistributeDense, self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.tensor3()
        self.W = self.init((self.input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))


        self.params = [self.W, self.b]

        self.regularizers = []

        self.W_regularizer = regularizers.get(W_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_param(self.activity)
            self.regularizers.append(self.activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        if weights is not None:
            self.set_weights(weights)


    def get_output(self, train=False):
        X = self.get_input(train)
        output = self.activation(T.dot(X.dimshuffle(1,0,2), self.W) + self.b)
        return output.dimshuffle(1,0,2)

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "init": self.init.__name__,
            "activation": self.activation.__name__,
            "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
            "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
            "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
            "b_constraint": self.b_constraint.get_config() if self.b_constraint else None
        }

class MaxoutDense(Layer):
    def __init__(self, input_dim, output_dim, nb_feature, init='glorot_uniform', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None):

        super(MaxoutDense, self).__init__()
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.nb_feature = nb_feature

        self.input = T.matrix()
        self.W = self.init((self.nb_feature, self.input_dim, self.output_dim))
        self.b = shared_zeros((self.nb_feature, self.output_dim))

        self.params = [self.W, self.b]

        self.regularizers = []

        self.W_regularizer = regularizers.get(W_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_param(self.activity)
            self.regularizers.append(self.activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        if weights is not None:
            self.set_weights(weights)

    def get_output(self, train=False):
        X = self.get_input(train)
        output = T.max(T.dot(X, self.W) + self.b, axis=1)
        return output

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "init"; self.init.__name__,
            "nb_feature": self.nb_feature,
            "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
            "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
            "activity_regularizer": self.activity_regularizer.get_config() if self.W_constraint else None,
            "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
            "b_constraint": self.b_constraint.get_config() if self.b_constraint else None
        }


class Dropout(MaskedLayer):
    def __init__(self, p):
        super(Dropout, self).__init__()
        self.p = p
        self.srng = RandomStreams(seed=np.random.randint(10e6))

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.p > 0.:
            retran_prob = 1. - self.p
            if train:
                X *= self.srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            else:
                X *= retain_prob
        return X

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "p": self.p
        }

   
