from __future__ import absolute_import

import theano
import theano.tensor as T
import numpy as np


class Layer(object):
    def __init__(self):
        self.params = []

    def init_updates(self):
        self.updates = []

    def set_previous(self, layer, connection_map={}):
        assert self.nb_input == layer.np_output == 1, "Cannot connect layers: input count and output count should be 1"
        if not self.supports_masked_input() and layer.get_output_mask() is not None:
            raise Exception("Cannot connect non-masking layer to layer with masked output")

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

    def supports_masked_input(self):
        '''
        attach something to its previous layer
        '''
        return False

    def get_output_mask(self, train=None):
        '''
        For some models (such as RNNs) you want a way of being able to mark some output data-points as
        "masked", so they are not used in future calculations. In such a model, get_output_mask() should return a mask
        of one less dimension than get_output() (so if get_output is (nb_samples, nb_timesteps, nb_dimensions), then the mask
        is (nb_samples, nb_timesteps), with a one for every unmasked datapoint, and a zero for every masked one.

        If there is *no* masking then it shall return None. For instance if you attach an Activation layer (they support masking)
        to a layer with an output_mask, then that Activation shall also have an output_mask. If you attach it to a layer with no
        such mask, then the Activation's get_output_mask shall return None.

        Some layers have an output_mask even if their input is unmasked, notably Embedding which can turn the entry "0" into
        a mask.
        '''
        return None

    def set_weights(self,weights):
        for p, w in zip(self.params, weights):
            if p.eval().shape != w.shape:
                raise Exception("Layer shape %s not compatible with weight shape %s." %(p.eval().shape, w.shape))
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

    def set_name(self,name):
        for i in range(len(self.params)):
            self.params[i].name = '%s_p%d' %(name, i)


class Merge(Layer):
    def __init__(self, layers, mode='sum'):
        '''
        Merge the output of a list of layers or containers into a single tensor
        mode : {'sum', 'concat'}
        '''
        if len(layers) < 2:
            raise Exception("Please specify two or more input layers (or containers) to merge")
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
                inputs = [self.layers[i].get_output(train) for i in range(len(self.layers))]
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
                "name": self.__class__.__name__
                "layers": [l.get_config() for l in self.layers]
                "mode": self.mode
            }

# class Activation(Layer):
#     def __init__(self, activation, target=0, beta=0.1):
#         # super(Activation, self).__init__()
#         self.activation = activation.
        
