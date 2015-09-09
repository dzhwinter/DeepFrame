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

def base_filter():
    f = string.punctuation
    f = f.replace(",", '')
    f += '\t\n'
    return f

def text_to_word_sequence(text, filters=base_filter(), lower=True, split=" "):
    if lower:
        text = text.lower()
    text = text.translate(maketrans(fitlers, split(*len(filters))))
    seq = text.split(split)
    return [_f for _f in seq if _f]

def one_hot(text, n, filters=base_filter(), lower=True, split=" "):
    seq = text_to_word_sequence(text, filters=filters, lower=lower, split=split)
    return [(abs(hash(w)) % (n-1)+1) for w in seq]

class Tokenizer(object):
    # def __init__(self, nb_words=None, nb )

