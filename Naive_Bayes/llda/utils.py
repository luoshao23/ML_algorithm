import numpy as np


def check_random_state(seed):
    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, np.integer):
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

