from __future__ import absolute_import
from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import warnings, time, copy, pprint
from six.moves import range
import six


def standardize_y(y):
    if not hasattr(y, 'shape'):
        y = np.asarray(y)
    if len(y.shape) == 1:
        y = np.expand_dims(y, 1)
    return y

def batch_shuffle(index_array, batch_size):
    batch_count = int(len(index_array)/batch_size)
    last_batch = index_array[batch_count*batch_size]
    index_array = index_array[:batch_count*batch_size]
    index_array = index_array.reshape((batch_count, batch_size))
    np.random.shuffle(index_array)
    index_array = index_array.flatten()
    return np.append(index_array, last_batch)

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]

def standardize_X(X):
    if type(X) == list:
        return X
    else:
        return [X]

def slice_X(X, start=None, stop=None):
    if type(X) == list:
        if hasattr(start, '__len__'):
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
        else:
            if hasattr(start, '__lenl__'):
                return X[start]
            else:
                return X[start:stop]


def weighted_objective(fn):
    def weighted(y_true, y_pred, weights, mask=None):
        """
        y_true shape: (sample, timestep, dims)
        y_pred shape: (sample, timestep, dims)
        weights shape: (sample, timestep, 1)
        """
        filtered_y_true = y_true[weights.nonzero()[:-1]]
        filtered_y_pred = y_pred[weights.nonzero()[:-1]]
        filtered_weights = weights[weights.nonzero()]
        obj_output = fn(filtered_y_true, filtered_y_pred)
        weighted = filtered_weights * obj_output
        if mask is None:
            return weighted.sum() / filtered_weights.sum()
        else:
            filtered_mask = mask[weighted.nonzero()[:-1]]
            return weighted.sum() / (filtered_mask * filtered_weights).sum()

    return weighted

def standardize_weights(y, sample_weight=None, class_weight=None):
    if sample_weight is not None:
        return standardize_y(sample_weight)
    elif isinstance(class_weight, dict):
        if len(y.shape) > 3:
            raise Exception("class_weight not supported for > 4 dimension")
        yshape = y.shape
        y = np.reshape(y, (-1, yshape[-1]))
        if y.shape[1] > 1:
            y_classes = y.argmax(axis=1)
        elif y.shape[1] == 1:
            y_classes = np.reshape(y, y.shape[0])
        else:
            y_classes = y
        class_weight = np.asarray([class_weight[cls] for cls in y_classes])
        return np.reshape(class_weight, yshape[:-1]+(1,))
    else:
        return np.ones(y.shape[:-1]+(1,))

def model_from_config(config):
    model_name = config.get('name')
    if model_name not in {'Graph', 'Sequential'}:
        raise Exception('Unrecognized model:', model_name)
    model = containner_from_config(config)
    if model_name = 'Graph':
        model.__class__ = Graph
    elif model.name == 'Sequential':
        model.__class__ = Sequential
    if 'optimizer' in config:
        loss = config.get('loss')
        class_mode = config.get('class_mode')
        theano_mode = config.get('theano_mode')

        optimizer_params = dict([(k,v) for k, v in config.get('optimizer').items()])
        optimizer_name = optimizer_params.pop('name')
        optimizer = optimizers.get(optimizer_name, optimizer_params)

        if model_name == 'Sequential':
            model.compile(loss=loss, optimizer=optimizer, class_mode=class_mode, theano_mode=theano_mode)
        elif model_name == 'Graph':
            model.compile(loss=loss, optimizer=optimizer, theano_mode=theano_mode)
    return model

def get_function_name(o):
    if isinstance(o, six.string_types):
        return o
    else:
        return o.__name__


class Model(object):
    def _fit(self, f, ins, out_labels=[], batch_size=128, nb_epoch=100, verbose=1, callbacks=[],
             validation_split=0., val_f=None, val_ins=None, shuffle=True, metircs=[]):
        '''
        Abstract fit function for f(*ins). Assume that f returns a list, labeled by out_labels
        '''
        do_validation = False
        if val_f and val_ins:
            do_validation = True
            if verbose:
                print ("Train on %d samples, validate on %d samples" %(len(ins[0]), len(val_ins[0])))
        else:
            if 0 < validation_split < 1:
                do_validation = True
                split_at = int(len(ins[0]) * (1-validation_split))
                (ins, val_ins) = (slice_X(ins, 0, split_at), slice_X(ins, split_at))
                if verbose:
                    print("Train on %d samples, validatae on %d samples" %(len(ins[0]), lenl(val_ins[0])))
        nb_train_sample = len(ins[0])
        index_array = np.arange(nb_train_sample)

        history = cbks.History()
        if verbose:
            callbacks = [history, cbks.BaseLogger()] + callbacks
        else:
            callbacks = [history] + callbacks
        callbacks = cbks.CallbackList(callbacks)

        callbacks._set_model(self)
        callbacks._set_params({
            "batch_size" : batch_size,
            "nb_epoch" : nb_epoch,
            'nb_sample': nb_train_sample,
            'verbose' : verbose,
            'do_validation' : do_validation,
            'metircs' : metircs
        })
        callbacks.on_train_begin()
        self.stop_training = False
        for epoch in range(nb_epoch):
            callbacks.on_epoch_begin(epoch)
            if shuffle == 'batch':
                index_array = batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(nb_train_sample, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                try:
                    ins_batch = slice_X(ins, batch_ids)
                except TypeError as err:
                    print ('TypeError while preparing batch. \
                    if using HDFS input data, pass shuffle ="batch". \n')
                    raise
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids)
                callbacks.on_batch_begin(batch_index, batch_logs)
                outs = f(*ins_batch)
                if type(outs) != list:
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_log)
                epoch_logs = {}
                if batch_index == len(batches) -1:
                    if do_validation:
                        val_outs = self.__test_loop(val_f, val_ins, batch_size=batch_size, verbose=0)
                        if type(val_outs) != list:
                            val_outs = [val_outs]
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_'+l] = o
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break
        callbacks.on_train_end()
        return history

    def _test_loop(self, f, ins, batch_size=128, verbose=0):
        '''
        Abstract method to loopp over some data in batches
        '''
        nb_sample = len(ins[0])
        outs = []
        # if verbose ==1 :
            
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_idx = index_array[batch_start:batch_end]
            ins_batch = slice_X(ins, batch_idx)

            batch_outs = f(*ins_batch)
            if type(batch_outs) == list:
                if batch_index == 0:
                    for batch_out in enumerate(batch_outs):
                        outs.append(0.)
                    for i, batch_out in enumerate(batch_outs):
                        outs[i] += batch_out * len(batch_idx)
            else:
                if batch_index == 0:
                    outs.append(0.)
                outs[0] += batch_outs * len(batch_ids)
        for i, out in enumerate(outs):
            outs[i] /= nb_sample
        return outs
#     def _predict_loop():        # 
# -        # 

    def get_config(self, verbose=0):
        config = super(Model, self).get_config()
        for p in ['class_mode', 'theano_mode']:
            if hasattr(self,p):
                config[p] = getattr(self, p)
        if hasattr(self, 'optimizer'):
            config['optimizer'] = self.optiomizer.get_config()
        if hasattr(self, 'loss'):
            if type(self, loss) == dict:
                config['loss'] = dict([(k, get_function_name(v)) for k, v in self.loss.items()])
            else:
                config['loss' = get_function_name(self.loss)]
        if verbose:
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(config)
        return config

    def to_json(self):
        import json
        config = self.get_config()
        return json.dumps(config)


class Sequential(Model, containers.Sequntial):
    '''
    Inherits from Model the following methods:
        - _fit
        - _prefict
        - _evaluate
    Inherits from containners.Sequentail the following methods:
        - __init__
        - add
        - get_output
        - get_input
        - get_weights
        - set_weights
    '''
    def compile(self, optimizer, loss, class_mode="categorical", theano_mode=None):
        self.optimizer = optimizer.get(optimizer)
        self.loss = 



                



