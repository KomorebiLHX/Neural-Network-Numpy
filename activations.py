import numpy as np


def sigmoid(input):
    return 1/(1+np.exp(-input))


def sigmoid_gradient(input):
    return sigmoid(input) * (1-sigmoid(input))


def relu(input):
    return np.maximum(0, input)


def relu_gradient(input):
    return input > 0


def softmax(input):
    return np.exp(input) / np.sum(np.exp(input))