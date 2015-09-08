import numpy as np
import random
import theano

from keras.datasets import mnist
from mykeras.callbacks import Callback
from mykeras.utils import generic_utils
from mykeras.utils import np_utils


# DrawActivations test

print("Runing DrawActivations test")
nb_classes = 10
batch_size = 128
nb_epoch = 10

max_train_smaples = 512
max_test_samples = 1
random.seed(1337)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 1, 28, 28)[:max_train_smaples]
X_train = X_train.astype("float32")
X_train /= 255

X_test = X_test.reshape(-1, 1, 28, 28)[:max_test_smaples]
X_test = X_test.astype("float32")
X_test /= 255

Y_train = np

