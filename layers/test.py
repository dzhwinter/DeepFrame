class A(object):
    def __init__(self):
        self.data = 1000

    def __call__(self, p):
        return p

    def get_data(self):
        return self.data

    @property
    def input(self):
        return self.get_data()

class B(A):
    def __init__(self):
        super(self,A).__init__()
        self.data = 2000

    def test(self, fun):
        def printB(self):
            print "print from inside B"

    def myprint(self):
        print "this print from B"
    # 
a = A()
print a
b = B()
b.test()

