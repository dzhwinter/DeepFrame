class A(object):
    def __init__(self):
        self.data = 1000

    def get_data(self):
        return self.data

    @property
    def input(self):
        return self.get_data()
