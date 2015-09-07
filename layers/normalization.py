from ..layers.cores import Layer
from ..utils.theano_utils import shared_ones, shared_zeros, ndim_tensor
from .. import initializations

import theano.tensor as T

class BatchNormalization(Layer):
    def __init__(self, input_shape, init='uniform', epsilon=1e-6, mode=0, momentum=0.9, weights=None):
        super(BatchNormalization, self).__init__()
        self.init = initializations.get(init)
        self.input_shape = input_shape
        self.epsilon = epsilon
        self.mode = mode
        self.momentum = momentum
        self.input = ndim_tensor(len(self.input_shape) + 1)

        self.gamma = self.init((self.input_shape))
        self.beta = shared_zeros(self.input_shape)

        self.params = [self.gamma, self.beta]
        if weights is not None:
            self.set_weights(weights)

    # def init_updates(self):
        self.running_mean = shared_zeros(self.input_shape)
        self.running_std = shared_ones((self.input_shape))
        X = self.get_input(train=True)
        m = X.mean(axis=0)
        std = T.mean((X - m) ** 2 + self.epsilon, axis = 0) ** 0.5
        mean_update = self.momentum * self.running_mean + (1-self.momentum) * m
        std_update = self.momentum * self.running_std + (1 - self.momentum) * std

        self.updates = [(self.running_mean, mean_update), (self.running_std, std_update)]

    def get_output(self, train):
        X = self.get_input(train)
        if self.mode == 0:
            X_normed = (X - self.running_mean) / (self.running_std + self.epsilon)
        elif self.mode == 1:
            m = X.mean(axis=-1, keepdims=True)
            std = X.std(axis=-1, keepdims=True)
            X_normed = (X - m) / (std+self.epsilon)

        out = self.gamma * X_normed + self.beta
        return out

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "input_shape": self.input_shape,
            "epsilon": self.epsilon,
            "mode": self.mode
        }
    
class LRN2D(Layer):
    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5):
        if n % 2== 0:
            raise NotImplementedError("LRN2D only support odd n.n provided" + str(n))
        super(LRN2D, self).__init__()
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        b, ch, r, c = X.shape
        half_n = self.n // 2
        extra_channels = T.alloc(0., b, ch+2*half_n, r, c)
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :], input_sqr)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        return X / scale

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "alpha": self.alpha,
            "k": self.k,
            "beta": self.beta,
            "n"; self.n
        }

