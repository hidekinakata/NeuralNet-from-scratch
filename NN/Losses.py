import numpy as np


class Loss:
    def loss(self, y, yhat):
        raise NotImplementedError

    def dLoss(self, y, yhat):
        raise NotImplementedError


class MSE(Loss):
    def loss(self, y, yhat):
        return (yhat - y) ** 2 / yhat.shape[0]

    def dLoss(self, y, yhat):
        return (yhat - y) * 2 / yhat.shape[0]
