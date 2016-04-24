# coding: utf-8
"""Information about mxnet."""
from __future__ import absolute_import
import os
import platform


lib = ctypes.cdll.LoadLibrary(lib_path[0])
__version__ = libinfo.__version__
_LIB = _load_lib()
