# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import sys
from six.moves import range, zip
from six import string_types
import string


if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

