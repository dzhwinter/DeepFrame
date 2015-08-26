import gzip
from .data_utils import get_file
import six.moves.cPickle
import sys

def load_data(path="mnist.pkl.gz"):
    # pathl = get_file(path, origiVn=)
    path = get_file(path, origin="https://s3.amazonaws.com/img-datasets/mnist.pkl.gz")

    if path.endswith("gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    if sys.version_info < (3,):
        data = six.moves.cPickle.load(f)
    else:
        data = six.moves.cPickle.load(f, encoding="bytes")
    f.close()

    return data

