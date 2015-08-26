from __future__ import absolute_import
from __future__ import print_function

import theano.tensor as T
from ..layers.cores import Layer, Merge
from ..utils.theano_utils import ndim_tensor
from six.moves import range


class Sequentail(Layer):
    '''
    Simple linear stack of layers

    inherited from layer:
    - get_params
    - get_output_mask
    - supports_masked_input
    '''
    def __init__(self, layers=[]):
        self.layers = []
        self.params = []
        self.regularizers = []
        self.constraints = []
        self.updates = []

        for layer in layers:
            self.add(layer)

    def set_previous(self, layer):
        self.layers[0].previous = layer

    def add(self, layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-1].set_previous(self.layers[-2])
            if not hasattr(self.layers[0], 'input'):
                self.set_input()
        layer.init_updates()

        params, regularizers, constraints, updates = layer.get_params()
        self.params += params
        self.regularizers += regularizers
        self.constraints += constraints
        self.updates += updates

    def get_output(self, train=False):
        return self.layers[-1].get_output(train)

    def set_output(self):
        for l in self.layers:
            if hasattr(l, 'input'):
                ndim = l.input.ndim
                self.layers[0].input = ndim_tensor(ndim)
                break

    def get_input(self, train=False):
        if not hasattr(self.layers[0], 'input'):
            self.set_input()
        return self.layers[0].get_input(train)

    @property
    def input(self):
        return self.get_input()

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.get_weights()
        return weights

    def set_weights(self, weights):
        for i in range(len(self.layers)):
            nb_param = len(self.layers[i].params)
            self.layer[i].set_weights(weightls[:nb_param])
            weights = weights[nb_param:]

    def get_config(self):
        return {
            "name" : self.__class__.__name__
            "layers" : [layer.get_config() for layer in self.layers]
        }


class Graph(Layer):
    '''
    NN Graph wiht arbitrary layer connections 
    Inherited from Layer:
        - get_params
        - get_output_mask
        - supports_masked_input
        - get_weights
        - set_weights

    '''

    def __init__(self):
        self.namespace = set()
        self.nodes = {}
        self.inputs = {}
        self.input_order = []
        self.outputs = {}
        self.output_order = []
        self.input_config = []
        self.output_config = []
        self.node_config = []

        self.params = []
        self.regularizers = []
        self.constraints = []
        self.updates = []

    @property
    def nb_input(self):
        return len(self.inputs)

    @property
    def nb_output(self):
        return len(self.outputs)

    def set_previous(self, layer, connection_map={}):
        if self.nb_input != layer.nb_output:
            raise Exception('Cannot connect layers: input count does not match output count')
        if self.nb_input == 1:
            self.inputs[self.input_order[0]].set_previous(layer)
        else:
            if not connection_map:
                raise Exception('Cannot attach multi-input layer: lno connection_map provided')
            for k, v in connection_map.items():
                if k in self.inputs and v in layer.outputs:
                    self.inputs[k].set_previous(layer.outputs[v])
                else:
                    raise Exception('Invalid connection map')

    def get_input(self, train=False):
        if len(self.inputs) == len(self.outputs) == 1:
            return self.inputs[self.input_order[0]].get_input(train)
        else:
            return dict([(k, v.get_input(train)) for k, v in self.input.items()])

    @property
    def input(self):
        return self.get_input()

    def get_output(self, train=False):
        if len(self.inputs) == len(self.outputs) == 1:
            return self.outputs[self.output_order[0]].get_output(train)
        else:
            return dict([(k, v.get_output(train)) for k, v in self.outputs.items()])

    def add_input(self, name, ndim=2, dtype='float'):
        if name in self.namespace:
            raise Exception('Duplicate node identifier :' + name)
        self.namespace.add(name)
        self.input_order.append(name)
        layer = Layer()
        if dtype == 'float':
            layer.input = ndim_tensor(ndim)
        else:
            if ndim == 2:
                layer.input = T.imatrix()
            else:
                raise Exception('Type "int" can only be used with ndim==2 (Embedding)')
        layer.input.name = name
        self.inputs[name] = layer
        self.input_config.append({'name':name, 'ndim': ndim, 'dtype': dtype})

    def add_node(self, layer, name, input=None, inputs=[], merge_mode='concat', create_output=False):
        if hasattr(layer, 'set_name'):
            layer.set_name(name)
        if name in self.namespace:
            raise Exception('Duplicate node identifier:'+name)
        if input:
            if input notl


    

