from scipy.special import expit as logistic_sigmoid
import numpy as np


def identity(X):
    return X

def logistic(X):
    return logistic_sigmoid(X, out=X)


def tanh(X):
    return np.tanh(X, out=X)


def relu(X):
    return np.clip(X, 0, np.finfo(X.dtype).max, out=X)


def deriv_identity(a, delta):
    """nothing"""


def deriv_logistic(a, delta):
    delta *= a
    delta *= (1.0 - a)



def deriv_tanh(a, delta):
    delta *= (1.0 - a**2)


def deriv_relu(a, delta):
    delta[a == 0] = 0

ACTIVATIONS = { 'identity': identity,'logistic': logistic,
               'tanh': tanh, 'relu': relu}

DERIVATIVES = {'identity':deriv_identity,'logistic': deriv_logistic,
               'tanh': deriv_tanh, 'relu': deriv_relu}
