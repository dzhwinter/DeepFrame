# coding: utf-8
"""Information about mxnet."""
from __future__ import absolute_import
import os
import platform

def find_lib_path():
    """Find MXNet dynamic library files.

    Returns
    -------
    lib_path : list(string)
        List of all found path to the libraries
    """
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    api_path = curr_path
    dll_path = [curr_path, api_path]
    dll_path = [os.path.join(p, 'libmxnet.so') for p in dll_path]
    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    if len(lib_path) == 0:
        raise RuntimeError('Cannot find the files.\n' +
                           'List of candidates:\n' + str('\n'.join(dll_path)))
    return lib_path

def _load_lib():
    """Load libary by searching possible path."""
    lib_path = libinfo.find_lib_path()
    lib = ctypes.cdll.LoadLibrary(lib_path[0])
    # DMatrix functions
    lib.MXGetLastError.restype = ctypes.c_char_p
    return lib

_LIB = _load_lib()
# current version
__version__ = "0.5.0"
