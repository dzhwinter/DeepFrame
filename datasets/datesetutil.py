import tarfile, inspect, osk
from six.moves.urllib.request import FancyURLopener

class URLopener(FancyURLopener):
    def http_error_default(self, url, fp, errcode, errmsg, headers):
        raise Exception('URL Fetch failure on {}: {} -- {}'.format(url, errcode, errmsg))


def get_file(path, origin):
    if os.path.exists(path):
        return path
    else:
        try:
            opener = URLopener()
            f = opener.open(origin)
            return f.read()
        except Exception, e:
            print "Download error", e
