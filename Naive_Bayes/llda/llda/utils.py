from __future__ import absolute_import
import numpy as np
from numbers import Integral

def check_random_state(seed):
    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, (np.integer, Integral)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed

    raise ValueError('{} cannont be used as a random seed.'.format(seed))


def matrix_to_lists(doc_word):
    """Convert a matrix of counts into a tuple of word and doc indices.

    Parameters
    ----------
    doc_word: array-like shape (D, V)

    Returns
    -------
    (WS, DS):
        WS[k] is the kth word in the corpus
        DS[k] is the document index for the kth word
    """

    ii, jj = np.nonzero(doc_word)
    ss = doc_word[ii, jj].astype(np.intc)

    DS = np.repeat(ii, ss).astype(np.intc)
    WS = np.repeat(jj, ss).astype(np.intc)

    return WS, DS


def lists_to_matrix(WS, DS):
    D = max(DS) + 1
    V = max(WS) + 1

    doc_word = np.zeros((D, V), dtype=np.intc)
    indices, counts = np.unique(list(zip(DS, WS)), axis=0, return_counts=True)
    doc_word[indices[:, 0], indices[:, 1]] += counts

    return doc_word

