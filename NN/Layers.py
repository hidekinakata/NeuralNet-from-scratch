import numpy as np

from NN.Activations import Relu, Sigmoid


class Layer:
    def __init__(self, neurons):
        self.neurons = neurons
        self.x = None
        pass

    def build(self, prev_layer):
        pass

    def forward(self, x):
        self.x = x
        return self.x

    def grad(self, d_output):
        n_params = self.x.shape[1]
        return d_output @ np.eye(n_params)


class InputLayer(Layer):
    def __init__(self, neurons):
        super().__init__(neurons)


class DenseLayer(Layer):
    def __init__(self, neurons, activation):
        super().__init__(neurons)
        self.weights = None
        self.bias = np.zeros((1, neurons))
        self.z = None
        self.a = None

        if type(activation) == str:
            if activation == 'sigmoid':
                self.activation_fn = Sigmoid()
            elif activation == 'relu':
                self.activation_fn = Relu()
            else:
                raise Exception("Invalid activation function")
        else:
            self.activation_fn = activation

    def build(self, prev_layer):
        self.weights = np.random.normal(size=(prev_layer, self.neurons),
                                        loc=0.0,
                                        scale=np.sqrt(2 / (prev_layer + self.neurons)), )

    def forward(self, x):
        self.x = x
        self.z = x @ self.weights + self.bias
        self.a = self.activation_fn.a(self.z)
        return self.a

    def grad(self, d_output):
        d_a = d_output * self.activation_fn.d(self.a)
        d_z = d_a @ self.weights.T
        d_w = self.x.T @ d_a
        d_b = d_a.sum(axis=0) * self.x.shape[0]

        return d_z, d_w, d_b

    def apply_grad(self, output_err, lr=1e-2):
        d_z, d_w, d_b = self.grad(output_err)
        self.weights -= lr * d_w
        self.bias -= lr * d_b
        return d_z

    def __call__(self, *args, **kwargs):
        self.forward(args[0])
        return self.a
