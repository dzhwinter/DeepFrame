from __future__ import absolute_import
import theano
import theano.tensor as T

from .utils.theano_utils import shared_zeros, shared_scalar, floatX
from .utils.generic_utils import get_from_module
from six.moves import zip


def clip_norml(g, c, n):
    if c > 0:
        g = T.switch(T.ge(n, c), g * c/n, g)
    return g


def kl_divergence(p, p_hat):
    return p_hat - p + p * T.log(p/p_hat)


class Optimizer(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.updates = []

    def get_state(self):
        return [u[0].get_value() for u in self.updates]

    def set_state(lself, value_list):
        assert len(self.updates) == len(value_list)
        for u, v in zip(self.updates, value_list):
            u[0].set_value(floatX(v))

    def get_updates(self, params , constraints, loss):
        raise NotImplemented

    def get_gradients(self, loss, params):
        grads = T.grad(loss, params)

        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = T.sqrt(sum([T.sum(g ** 2) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]

        return grads

    def get_config(self):
        return {"name": self.__class__.__name__}


class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0, decay=0., nesterov=Falsem, *args, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = shared_scalar(0)
        self.lr = shared_scalar(lr)
        self.momentum = shared_scalar(momentum)


    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        self.updates = [(self.iterations, self.iterations + 1.)]

        for p, g, c in zip(params, grads, constraints):
            m = shared_zeros(p.get_value().shape)
            v = self.moementum *m - lr * g
            self.updates.append((m,v))

            if self.nesterov:
                new_p = p + self.momentum + v - lr*g
            else:
                new_p = P + v
            self.updates.append((p, c(new_p)))
        return self.updates

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "lr": float(self.lr.get_value()),
            "momentum": float(self.momentum.get_value()),
            "decay": float(self.decay.get_value()),
            "nesterov": self.nesterov
        }


class RMSprop(Optimizer):
    def __init__(self, lr=0.01, rho=0.9, epsilon=1e-6, *args, **kwargs):
        super(RMSprop, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = shared_scalar(lr)
        self.rho = shared_scalar(rho)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        accomulators = [shared_zeros(p.get_value().shape) for p in params]
        self.updates = []

        for p, g, a, c in zip(params, grads, accomulators, constraints):
            new_a = self.rho *a + (1 - self.rho) * g ** 2
            self.updates.append((a, new_a))
            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            self.updates.append((p, c(new_p)))
        return self.updates

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "lr" : float(self.lr.get_value()),
            "rho": float(self.rho.get_value()),
            "epsilon": self.epsilon
        }


# give some alias
sgd = SGD
rmsprop = RMSprop

def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'optimizer', instantiate=True, kwargs=kwargs)
