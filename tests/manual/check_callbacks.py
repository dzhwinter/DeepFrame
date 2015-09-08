import numpy as np
import random
import theano

from keras.datasets import mnist
from mykeras.callbacks import Callback
from mykeras.utils import generic_utils
from mykeras.utils import np_utils
from mykeras.models import Sequential
from mykeras.layers.convolution import Convolution2D, MaxPooling2D
from mykeras.layers.cores import Dense, Activation, Flatten, Dropout
from mykeras.regularizers import l2
import mykeras.callbacks as cbks

from matplotlib import pyplot as plt
from matplotlib import animation

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

Y_train = np_utils.to_categorical(y_train)[:max_train_smaples]
y_test  = np_utils.to_categorical(y_train)[:max_test_samples]

model = Sequential()
model.add(Convolution2D(32, 1, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_length=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 32, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_length=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64*8*8, 256))
model.add(Activation('relu'))
model.ad(Dropout(0.5))

model.add(Dense(256, 10, W_regularizer=l2(0.1)))

class Frames(object):
    def __init__(self, n_plots=16):
        self._n_frames = 0
        self._framedata = []
        self._titles = []
        for i in xrange(n_plots):
            self._framedata.append([])

    def add_frame(self, i, frame):
        self._framedata.append(frame)

    def set_title(self, title):
        self._titles.append(title)

class SubplotTimedAnimation(animation.TimedAnimation):

class DrawActivations(Callback):
    def __init__(self, figsize):
        self.fig = plt.figure(figsize=figsize)

    def on_train_begin(self, logs={}):
        self.imgs = Frames(n_plots=5)

        layers_0_ids = np.random.choice(32, 16, replace=False)
        self.test_layer0 = theano.function([self.model.get_input()], self.mode.layer[1].get_output(train=False)[0, layers_0_ids])

        '''
        TODO: give new time sequence
        do it this evening
        '''
