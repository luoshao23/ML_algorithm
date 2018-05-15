import numpy as np
import warnings


def _num_samples(x):
    if hasattr(x, 'fit') and callable(x.fit):
        raise TypeError(
            "Expected sequence or array-like, got estimator %s" % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError(
                "Expected sequence or array-like, got %s" % type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton")
        return x.shape[0]
    else:
        return len(x)


def check_consistent_length(*arrays):
    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Length not equal!")


def column_or_1d(y, warn=False):
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn("column vector")
        return np.ravel(y)
    raise ValueError("bad input")


def check_X_y(X, y):
    check_consistent_length(X, y)
    return X, y
