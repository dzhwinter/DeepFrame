from ..callbacks import Callback


class RemoteMonitor(Callback):
    def __init__(self, root='http://localhost:9000'):
        self.root = root

    def on_epoch_begin(self, epoch, logs={}):
        self.seen = 0
        self.totals = {}

    def on_epoch_end(self, epoch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size
        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] += v * batch_size
