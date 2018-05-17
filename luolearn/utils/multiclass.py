from collections import Sequence

import numpy as np
from scipy.sparse.base import spmatrix

from ..externals.six import string_types


def is_multilabel(y):
    if hasattr(y, '__array__'):
        y = np.asarray(y)
    if not (hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1):
        return False


def type_of_target(y):
    valid = ((isinstance(y, (spmatrix, Sequence)) or hasattr(
        y, '__array__')) and not isinstance(y, string_types))

    if not valid:
        raise ValueError('Expected array-like')

    sparseseries = (y.__class__.__name__ == 'SparseSeries')
    if sparseseries:
        raise ValueError("y cannot be class 'SparseSeries'.")

    if is_multilabel(y):
        return 'multilabel-indicator'

    try:
        y = np.asarray(y)
    except ValueError:
        return 'unknown'

    try:
        if (not hasattr(y[0], '__array__') and isinstance(
                y[0], Sequence) and not isinstance(y[0], string_types)):
            raise ValueError(
                'You appear to be using a legacy multi-label data')
    except IndexError:
        pass

    if y.ndim > 2 or (y.dtype == object and len(y) and
                      not isinstance(y.flat[0], string_types)):
        return 'unknown'  # [[[1, 2]]] or [obj_1] and not ["label_1"]

    if y.ndim == 2 and y.shape[1] == 0:
        return 'unknown'  # [[]]

    if y.ndim == 2 and y.shape[1] > 1:
        suffix = "-multioutput"  # [[1, 2], [1, 2]]
    else:
        suffix = ""  # [1, 2, 3] or [[1], [2], [3]]

    # check float and contains non-integer float values
    if y.dtype.kind == 'f' and np.any(y != y.astype(int)):
        # [.1, .2, 3] or [[.1, .2, 3]] or [[1., .2]] and not [1., 2., 3.]
        return 'continuous' + suffix

    if (len(np.unique(y)) > 2) or (y.ndim >= 2 and len(y[0]) > 1):
        return 'multiclass' + suffix  # [1, 2, 3] or [[1., 2., 3]] or [[1, 2]]
    else:
        return 'binary'  # [1, 2] or [["a"], ["b"]]
