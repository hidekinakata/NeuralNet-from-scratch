import numpy as np


class Activation:
    def a(self, x):
        raise NotImplementedError()

    def d(self, x):
        raise NotImplementedError()


class Sigmoid(Activation):
    def a(self, x):
        return 1 / (1 + np.exp(-x))

    def d(self, x):
        sig = self.a(x)
        return sig*(1 - sig)


class Relu(Activation):
    def a(self, x):
        return np.maximum(0, x)

    def d(self, x):
        x[x <= 0] = 1e-9
        x[x > 0] = 1
        return x
