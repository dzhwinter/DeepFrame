from __future__ import absolute_import
from __future__ import print_function

import tarfile, inspect, os
from six.moves.urllib.request import FancyURLopener

class ParanoidURLopener(FancyURLopener):
    def http_error_default(self, url, fp, errcode, errmsg, headers):
        raise Exception('URL Fetch failure on {}: {} -- {}'.format(url, errcode, errmsg))

def get_file(fname, origin, untar =False):
    datadir = os.path.expanduser(os.path.join('~','.keras','datasets'))
    if not os.path.exists(datadir):
        print "you need to mkdir :" + datadir

    # if untar
