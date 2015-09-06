import unittest
import numpy as np

from numpy.testing import assert_allclose
import theano

from mykeras.layers import cores


class TestLayerBase(unittest.TestCase):
    def test_input_output(self):
        nb_samples = 10
        input_dim = 5
        layer = cores.Layer()

        for train in [True, False]:
            self.assertRaises(AttributeError, layer.get_input, train)
            self.assertRaises(AttributeError, layer.get_output, train)

        input = np.ones((nb_samples, input_dim))
        layer.input = theano.shared(value=input)
        for train in [True, False]:
            assert_allclose(layer.get_input(train).eval(), input)
            assert_allclose(layer.get_output(train).eval(), input)

    def test_connections(self):
        nb_samples = 10
        input_dim = 5
        layer1 = cores.Layer()
        layer2 = cores.Layer()

        input = np.ones((nb_samples, input_dim))
        layer1.input = theano.shared(value=input)

        for train in [True ,False]:
            self.assertRaises(AttributeError, layer2.get_input, train)

        layer2.set_previous(layer1)
        for train in [True, False]:
            assert_allclose(layer2.get_input(train).eval(), input)
            assert_allclose(layer2.get_output(train).eval(), input)
