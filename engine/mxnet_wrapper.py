#coding:utf8
from __future__ import absolute_import
import types
import operator
from ..third_party.mxnet.libmxnet
import mxnet
from mxnet import ndarray
from mxnet.ndarray import NDArray

def wrap_namespace(ns, reg, prim_wrapper):
    """Wrap namespace into a reg.

    :param ns: Namespace from which functions are to be wrapped.
    :param reg: Registry into which functions are to be wrapped.
    :param prim_wrapper: Wrapper to convert a raw function to primitive
    """
    unchanged_types = {float, int, type(None), type}
    function_types = {types.FunctionType, types.BuiltinFunctionType}

    for name, obj in ns.items():
        if type(obj) in function_types:
            prim = prim_wrapper(obj)
            reg.register(name, prim)


def unbroadcast(ans, x, gradfun):
    if isinstance(ans, NDArray) and isinstance(x, NDArray):
        padded_shape = (1,) * (len(ans.shape) - len(x.shape)) + x.shape
        def newgradfun(g):
            gg = gradfun(g)
            for axis, (i, j) in enumerate(zip(g.shape, padded_shape)):
                if i != j:
                    gg = ndarray.sum(gg, axis=axis, keepdims=True)
            if gg.shape != x.shape:
                gg = gg.reshape(x.shape)
            return gg
        return newgradfun
    elif isinstance(ans, NDArray): # x is numerical value
        def newgradfun(g):
            gg = gradfun(g)
            return ndarray.sum(gg)
    else: # both ans and x are numerical value
        return gradfun

def register_primitives(reg, prim_wrapper):
    mxnet_wrapper.wrap_namespace(mxnet.ndarray.__dict__, reg, prim_wrapper)

def gen_sum_grad(ans, x, axis, keepdims):
    xshape = list(x.shape)
    if axis is None:
        return lambda g: ndarray.full(x.shape, g, x.context)
    if type(axis) is int:
        axis = [axis]
    elif type(axis) is tuple:
        axis = list(axis)
    for a in axis:
        xshape[a] = 1
    def sum_grad(g):
        return ndarray.zeros(x.shape, ctx=g.context) + g.reshape(tuple(xshape))
    return sum_grad

def def_grads(reg, prims):
    def identity(x):
        return x
    # dot
    prims('dot').def_grad(lambda ans, a, b: lambda g: ndarray.dot(g, b.T))
    prims('dot').def_grad(lambda ans, a, b: lambda g: ndarray.dot(a.T, g), argnum=1)
    # non-linear
    #prims.tanh.def_grad(lambda ans, x: lambda g: g / np.cosh(x) ** 2)
    prims('exp').def_grad(lambda ans, x: lambda g: g * ans)
    prims('log').def_grad(lambda ans, x: lambda g: g / x)
    # reduce
    prims('sum').def_grad(lambda ans, x, axis=None, keepdims=False: gen_sum_grad(ans, x, axis, keepdims))
    # + - * /
    prims('multiply').def_grad(lambda ans, x, y: unbroadcast(ans, x, lambda g: g * y))
    prims('multiply').def_grad(lambda ans, x, y: unbroadcast(ans, y, lambda g: x * g), argnum=1)
    prims('add').def_grad(lambda ans, x, y: unbroadcast(ans, x, identity))
    prims('add').def_grad(lambda ans, x, y: unbroadcast(ans, y, identity), argnum=1)
    prims('subtract').def_grad(lambda ans, x, y: unbroadcast(ans, x, identity))
    prims('subtract').def_grad(lambda ans, x, y: unbroadcast(ans, y, operator.neg), argnum=1)
    prims('divide').def_grad(lambda ans, x, y: unbroadcast(ans, x, lambda g: g / y))
    prims('divide').def_grad(
            lambda ans, x, y: unbroadcast(ans, y, lambda g: - g * x / (y * y)),
            argnum=1)
    prims('true_divide').def_grad(lambda ans, x, y: unbroadcast(ans, x, lambda g: g / y))
    prims('true_divide').def_grad(
            lambda ans, x, y: unbroadcast(ans, y, lambda g: - g * x / (y * y)),
            argnum=1)
    # power
    #prims.power.def_grad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g * y * x ** (y - 1)))
    #prims.power.def_grad(lambda ans, x, y : unbroadcast(ans, y, lambda g : g * ndarray.log(x) * x ** y), argnum=1)
    # mod
    #prims.mod.def_grad(lambda ans, x, y : unbroadcast(ans, x, identity))
    #prims.mod.def_grad(lambda ans, x, y : unbroadcast(ans, y, lambda g : - g * ndarray.floor(x/y)), argnum=1)
    # negate
    prims('negative').def_grad(lambda ans, x: operator.neg)
    prims('abs').def_grad(lambda ans, x: lambda g: mxnet.nd.sign(x) * g)
    prims('sign').def_grad_zero()
    prims('round').def_grad_zero()
    prims('ceil').def_grad_zero()
    prims('floor').def_grad_zero()
    prims('sqrt').def_grad(lambda ans, x: lambda g: g * 0.5 / mxnet.nd.sqrt(x))
    prims('sin').def_grad(lambda ans, x: lambda g: g * mxnet.nd.cos(x))
    prims('cos').def_grad(lambda ans, x: lambda g: -g * mxnet.nd.sin(x))
