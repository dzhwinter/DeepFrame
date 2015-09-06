from __future__ import absolute_import

import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn

from .. import activations, initializations, regularizers, constraints
from ..utils.theano_utils import shared_zeros
from ..layers.cores import Layer

class Convolution1D(Layer):
    def __init__(self, input_dim, nb_filter, filter_length,
                 init='uniform', activation='linear', weights=None,
                 border_mode='valid', subsample_length=1,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None):
        if border_mode not in ['valid', 'full', 'same'] :
            raise Exception('Invalid border mode for Convolution1D:', border_mode)

        super(Convolution1D, self).__init__()
        self.nb_filter = nb_filter
        self.input_dim = input_dim
        self.filter_length = filter_length
        self.subsample_length = subsample_length
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.subsample = (1, subsample_length)
        self.border_mode = border_mode

        self.input = T.tensor3()
        self.W_shape = (nb_filter, input_dim, filter_length, 1)
        self.W = self.init(self.W_shape)
        self.b = shared_zeros((nb_filter, ))

        self.params = [self.W, self.b]

        self.regularizers = []

        self.W_regularizer = regularizers.get(W_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(b_regularizer)
            self.regularizers.append(self.b_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint) 
        self.constraints = [self.W_constraint, self.b_constraint]

        if weights is not None:
            self.set_weights(weights)

    def get_output(self, train):
        X = self.get_input(train)
        X = T.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1)).dimshuffle(0,2,1,3)

        border_mode = self.border_mode
        if border_mode == 'same':
            border_mode = 'full'
        conv_out = T.nnet.conv.conv2d(X, self.W, border_mode=border_mode, subsample=self.subsample)
        if self.border_mode == 'same':
            shift_x = (self.filter_length - 1) // 2
            conv_out = conv_out[:, :, shift_x:X.shape[2]+shift_x, :]

        output = self.activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        output = T.reshape(output, (output.shape[0], output.shape[1], output.shape[2])).dimshuffle(0,2,1, 2)
        return output

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "input_dim": self.input_dim,
            "nb_filter": self.nb_filter,
            "filter_length": self.filter_length,
            "init": self.init.__name__,
            "activation": self.activation.__name__,
            "border_mode": self.border_mode,
            "subsample_length": self.subsample_length,
            "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
            "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
            "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
            "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
            "b_constraint": self.b_constraint.get_config() if self.b_constraint else None
        }

class Convolution2D(Layer):
    def __init__(self, nb_filter, stack_size, nb_row, nb_col,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1,1),
                 W_regularizer=None, b_regularizer=None, activaty_regularizer=None,
                 W_constraint=None, b_constraint=None):

        if border_mode not {'valid', 'full', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:' , border_mode)

        super(Convolution2D, self).__init__()
        self.init = initializations.get(init)
        self.activations = activations.get(activation)
        self.subsample = subsample
        self.border_mode = border_mode
        self.nb_filter = nb_filter
        self.stack_size = stack_size

        self.nb_row = nb_row
        self.nb_col = nb_col

        self.input = T.tensor4()
        self.W_shape = (nb_filter, stack_size, nb_row, nb_col)
        self.W = self.init(self.W_shape)
        self.b = shared_zeros((nb_filter, ))

        self.params = [self,W, self,b]

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

    def get_output(self, train):
        X = self.get_input(train)
        border_mode = self.border_mode
        if dnn.dnn_available() and theano.config.device[:3] == 'gpu':
            if border_mode == 'same':
                assert(self.subsample == (1,1))
                pad_x = (self.nb_row - self.subsample[0]) // 2
                pad_y = (self.nb_col - self.subsample[1]) // 2
                conv_out = dnn.dnn_conv(img=X,
                                        kerns=self.W,
                                        border_mode=(pad_x, pad_y))
            else:
                conv_out = dnn.dnn_conv(img=X,
                                        kerns=self.W,
                                        border_mode=border_mode,
                                        subsample=self.subsample)

        else:
            if border_mode == 'same':
                border_mode = 'full'
            conv_out = T.nnet.conv.conv2d(X, self.W,
                                          border_mode=border_mode,
                                          subsample=self.subsample)

            if self.border_mode == 'same':
                shift_x = (self.nb_row - 1) // 2
                shift_y = (self.nb_col -1 )// 2
                conv_out = conv_out[:, :, shift_x:X.shape[2]+shift_x, shift_y:X.shape[3]+shift_y]

        return self.activations(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))


    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "nb_filter": self.nb_filter,
            "stack_size": self.stack_size,
            "nb_row": self.nb_row,
            "nb_col": self.nb_col,
            "init": self.init.__name__,
            "activation": self.activation.__name__,
            "border_mode": self.border_mode,
            "subsample": self.subsample,
            "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
            "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
            "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
            "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
            "b_constraint": self.b_constraint.get_config() if self.b_constraint else None
        }


class MaxPooling1D(Layer):
    def __init__(self, pool_length =2, stride=None, ignore_border=True):
        super(MaxPooling1D, self).__init__()
        self.pool_length = pool_length
        self.stride = stride
        if self.stride:
            self.st = (self.stride, 1)
        else:
            self.st = None

        self.input = T.tensor3()
        self.poolsize = (pool_length, 1)
        self.ignore_border = ignore_border

    def get_output(self, train):
        X = self.get_input(train)
        X = T.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1)).dimshuffle(0,2,1,3)
        output = T.signal.downsample.max_pool_2d(X, ds=self.poolsize, st=self.st, ignore_border=self.ignore_border)
        output = output.dimshuffle(0,2,1,3)
        return T.reshape(output, (output.shape[0], output.shape[1], output.shape[2]))

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "pool_length", self.pool_length,
            "stride": self.stride,
            "ignore_border": self.ignore_border
        }

class MaxPooling2D(Layer):
    def __init__(self, pool_length=2, stride=None, ignore_border=True):
        super(MaxPooling2D, self).__init__()
        self.pool_length = pool_length
        self.stride = stride
        if self.stride :
            self.st = (self.stride,1)
        else:
            self.st = None

        self.input = T.tensor3()
        self.poolsize = (pool_length, 1)
        self.stride = stride
        if self.stride:
            self.st = (self.stride, 1)
        else:
            self.st = None

        self.input = T.tensor3()
        self.poolsize = (pool_length, 1)
        self.ignore_border = ignore_border

    def get_output(self, train):
        X = self.get_input(train)
        X = T.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1)).dimshuffle(0,2,1,3)
        output = T.signal.downsample(0,2,1,3)
        return T.reshape(output, (output.shape[0], output.shape[1], output.shape[2]))

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "stride": self.stride,
            "pool_length", self.pool_length,
            "ignore_border": self.ignore_border
        }
    

    

