import numpy as np


def label_to_classes(y_data):
    converted = np.zeros((len(y_data), np.max(y_data) - np.min(y_data) + 1))
    for i in range(len(y_data)):
        converted[i, y_data[i]] = 1

    return converted


def accuracy(y, y_hat):
    test = (np.argmax(y, axis=1) == np.argmax(y_hat, axis=1))
    return test.sum()*100 / len(y)
