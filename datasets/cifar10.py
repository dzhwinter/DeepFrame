#coding:utf8
import os, sys, pickle, shutil
import logging, cPickle

"""
CIFAR-10 Image classification dataset
http://www.cs.toronto.edu/~kriz/cifar.html
http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

"""
URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
logger = logging.getLogger(__name__)

def load_data(path="cifarl10.pkl.gz"):
    path = get_file(path, origin=URL)
    if path.endswith("gz"):
        self = CIFAR10()
        self.build_meta()
        # f = gzip.open(path, 'rb')
        return self._pixels
    else:
        f = open(path, 'rb')
    if sys.version_info < (3,):
        data = six.moves.cPickle.load(f)
    else:
        data = six.moves.cPickle.load(f, encoding="bytes")
    f.close()

    return data

class CIFAR10(object):
    def __init__(self):
        self.meta_const = dict(
                image = dict(
                    shape = (32, 32, 3),
                    dtype = 'uint8',
                    )
                )
        self.descr = dict(
                n_classes = 10,
                )

    def __get_meta(self):
        try:
            return self._meta
        except AttributeError:
            self.fetch(download_if_missing=self.DOWNLOAD_IF_MISSING)
            self._meta = self.build_meta()
            return self._meta
    meta = property(__get_meta)

    def build_meta(self):
        try:
            self._pixels
        except AttributeError:
            # load data into class attributes _pixels and _labels
            pixels = np.zeros((60000, 32, 32, 3), dtype='uint8')
            labels = np.zeros(60000, dtype='int32')
            fnames = ['data_batch_%i'%i for i in range(1,6)]
            fnames.append('test_batch')

            # load train and validation data
            n_loaded = 0
            for i, fname in enumerate(fnames):
                data = self.unpickle(fname)
                assert data['data'].dtype == np.uint8
                def futz(X):
                    return X.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
                pixels[n_loaded:n_loaded + 10000] = futz(data['data'])
                labels[n_loaded:n_loaded + 10000] = data['labels']
                n_loaded += 10000
            assert n_loaded == len(labels)
            CIFAR10._pixels = pixels
            CIFAR10._labels = labels
            assert LABELS == self.unpickle('batches.meta')['label_names']
        meta = [dict(
                    id=i,
                    split='train' if i < 50000 else 'test',
                    label=LABELS[l])
                for i,l in enumerate(self._labels)]
        return meta

    def unpickle(self, basename):
        fo = open(fname, 'rb')
        data = cPickle.load(fo)
        fo.close()
        return data

    def classification_task(self):
        #XXX: use .meta
        y = self._labels
        X = self.latent_structure_task()
        return X, y

    def latent_structure_task(self):
        return self._pixels.reshape((60000, 3072)).astype('float32') / 255



def main_show():
    from utils.glviewer import glumpy_viewer, glumpy
    Y = [m['label'] for m in self.meta]
    glumpy_viewer(
            img_array=CIFAR10._pixels,
            arrays_to_print=[Y],
            cmap=glumpy.colormap.Grey,
            window_shape=(32 * 2, 32 * 2))


