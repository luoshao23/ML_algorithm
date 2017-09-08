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


def softmax(X):
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]

    return X


def deriv_identity(a, delta):
    """nothing"""


def deriv_logistic(a, delta):
    delta *= a
    delta *= (1.0 - a)


def deriv_tanh(a, delta):
    delta *= (1.0 - a**2)


def deriv_relu(a, delta):
    delta[a <= 0] = 0


def squared_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() / 2


def log_loss(y_true, y_prob):

    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    return -np.sum(y_true * np.log(y_prob)) / y_prob.shape[0]


def binary_log_loss(y_true, y_prob):

    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

    return -np.sum(y_true * np.log(y_prob) +
                   (1 - y_true) * np.log(1 - y_prob)) / y_prob.shape[0]


ACTIVATIONS = {'identity': identity, 'logistic': logistic,
               'tanh': tanh, 'relu': relu, 'softmax': softmax}

DERIVATIVES = {'identity': deriv_identity, 'logistic': deriv_logistic,
               'tanh': deriv_tanh, 'relu': deriv_relu}

LOSS_FUNCTIONS = {'squared_loss': squared_loss, 'log_loss': log_loss,
                  'binary_log_loss': binary_log_loss}
