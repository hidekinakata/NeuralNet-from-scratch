import numpy as np

from NN.utils import accuracy


class SequentialNeuralNet:
    def __init__(self, layers: list):
        self.built = False
        self.layers = layers
        self.loss = []
        self.loss_fn = None

    def build(self, loss_fn):
        self.built = True
        self.loss_fn = loss_fn
        for li in range(1, len(self.layers)):
            self.layers[li].build(self.layers[li - 1].neurons)

    def fit(self, x, y, epochs=10, lr=1e-3, showAccuracy=False):
        if len(y.shape) == 1:
            y = y.reshape((len(y), 1))
        self.loss = []

        for e in range(epochs):
            forw = self.predict(x)
            error = self.loss_fn.dLoss(y, forw)
            for l in range(len(self.layers) - 1, 0, -1):
                error = self.layers[l].apply_grad(error, lr)
                # print(error)

            forw = self.predict(x)
            self.loss.append(self.loss_fn.loss(y, forw).mean())
            if e % (epochs // 100 + 1) == 0:
                back = '\b'
                status = f'Epoch {e}/{epochs} - loss: {self.loss_fn.loss(y, forw).mean():.4e}'
                if showAccuracy:
                    status += f" Accuracy: {accuracy(y, forw):.2f}"
                print(f'{back * len(status)}{status}', end='', flush=True)

    def predict(self, x):
        self.layers[0].a = x
        for li in range(1, len(self.layers)):
            self.layers[li](self.layers[li - 1].a)

        return self.layers[-1].a

    def print_weights(self):
        for li in range(1, len(self.layers)):
            print(self.layers[li].weights)
