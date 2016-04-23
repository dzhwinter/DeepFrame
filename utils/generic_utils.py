from __future__ import absolute_import
import numpy as np
import time
import sys
import six


def get_from_module(identifier, module_params, module_name, instantiate=False, kwargs=None):
    if isinstance(identifier, six.string_types):
        res = module_params.get(identifier)
        if not res:
            raise Exception('Invalid ' + str(module_name) + ': '+str(identifier))
        if instantiate and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res
    return identifier


def make_tuple(*args):
    return args


def printv(v, prefix=''):
    if type(v) == dict:
        if 'name' in v:
            print(prefix + '#' + v['name'])
            del v['name']
        prefix += '...'
        for nk, nv in v.items():
            if type(nv) in [dict, list]:
                print (prefix+nk+':')
                printv(nv, prefix)
            else:
                print(prefix + nk + ':' + str(nv))
    elif type(v) == list:
        prefix += '...'
        for i, nv in enumerate(v):
            print(prefix + '#' + str(i))
            printv(nv, prefix)
    else:
        prefix += '...'
        print(prefix + str(v))
