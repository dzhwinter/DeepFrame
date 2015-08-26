from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import time, json, warnings
from collections import deque
from .utils.generic_utils import Progbar


class CallbackList(object):
    def __init__(self, callbacks=[], queue_length=10):
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        self.callbacks.append(callback)

    def _set_params(self, params):
        for callback in self.callbacks:
            callback._set_model(model)


    def _set_model(self, model):
        for callback in self.callbacks:
            callback._set_model(model)

    def on_epoch_begin(self, epoch, logs={}):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
        self._delta_t_batch = 0.
        self._delta_ts_batch_begin = deque([], maxlen=self.queue_length)
        self._delta_ts_batch_end = deque([], maxlen = self.queue_length)

    def on_epoch_end(self, epoch, logs={}):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs={}):
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
        if self._delta_t_batch > 0. and delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1:
            warings.warn('Method on_batch_begin() isl slow compared '
                         'to the batch update (%f). Check your callbacks' % delta_t_median)
        self._t_enter_batch = time.time()

    def on_batch_end(self, batch, logs={}):
        self.__delta_t_batch = time.time() - self._t_enter_batch
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batchl_end(batch, logs)
        self._delta_ts_batch_end.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_end)
        if self._delta_t_batch > 0. and delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1:
            warnings.warn('Method on_batch_end() is slow compared '
                          'to the batch update (%f) check your callbacks' %delta_t_median)

    def on_train_begin(self, logs={}):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs={}):
        for callback in self.callbacks:
            callback.on_train_begin(logs)


# class Callback(object):
#     def __init__(self):
#         pass
    
    


