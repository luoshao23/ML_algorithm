import warnings

import numpy as np


def empirical_covariance(X, assume_centered=False):
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    if X.shape[0] == 1:
        warnings.warn('Only one sample available')

    if assume_centered:
        covariance = np.dot(X.T, X) / X.shape[0]
    else:
        covariance = np.cov(X.T, bias=True)
