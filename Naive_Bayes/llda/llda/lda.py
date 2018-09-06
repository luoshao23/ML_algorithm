from __future__ import absolute_import
import numpy as np

import llda.utils
import llda._lda


class LDA(object):

    def __init__(self, n_topics, n_iter=2000, alpha=0.1, eta=0.01, random_state=None, refresh=10):
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.alpha = alpha
        self.eta = eta

        self.random_state = random_state
        self.refresh = refresh

        rng = llda.utils.check_random_state(random_state)
        self._rands = rng.rand(1024**2 // 8)

    def fit(self, X, y=None):
        """Set docstring here.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
        y=None:

        Returns
        -------
        self
        """
        self._fit(X)
        return self

    def transform(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = np.atleast_2d(X)
        self._fit(X)
        return self.doc_topic_

    def _fit(self, X):
        # check random state
        random_state = llda.utils.check_random_state(self.random_state)
        rands = self._rands.copy()
        # generate the initial distribution for Mdz, Mzw and Az
        self._initialize(X)
        for it in range(self.n_iter):
            # FIXME: using numpy.roll with a random shift might be faster
            random_state.shuffle(rands)
            if it % self.refresh == 0:
                # calculate the loglikelihood
                ll = self.loglikelihood()
                self.loglikelihoods_.append(ll)
            self._sample_topics(rands)
        ll = self.loglikelihood()
        self.components_ = (self.nzw_ + self.eta).astype(float)
        self.components_ /= np.sum(self.components_, axis=1, keepdims=True)
        self.topic_word_ = self.components_
        self.doc_topic_ = (self.ndz_ + self.alpha).astype(float)
        self.doc_topic_ /= np.sum(self.doc_topic_, axis=1, keepdims=True)

        del self.WS
        del self.DS
        del self.ZS
        return self

    def _initialize(self, X):
        D, W = X.shape
        N = int(X.sum())
        n_topics = self.n_topics
        n_iter = self.n_iter

        # topics - word matrix
        self.nzw_ = nzw_ = np.zeros((n_topics, W), dtype=np.intc)
        # document - topics matrix
        self.ndz_ = ndz_ = np.zeros((D, n_topics), dtype=np.intc)
        # topics array
        self.nz_ = nz_ = np.zeros(n_topics, dtype=np.intc)

        self.WS, self.DS = WS, DS = llda.utils.matrix_to_lists(
            X)  # the kth word in the corpus, the document index for the kth word
        np.testing.assert_equal(N, len(WS))
        self.ZS = ZS = np.empty_like(self.WS, dtype=np.intc) # the topic for the word
        # initialize the distribution for Mdz, Mzw and Az
        for i in range(N):
            w, d = WS[i], DS[i]
            z_new = i % n_topics
            ZS[i] = z_new
            ndz_[d, z_new] += 1
            nzw_[z_new, w] += 1
            nz_[z_new] += 1
        self.loglikelihoods_ = []

    def loglikelihood(self):
        nzw, ndz, nz = self.nzw_, self.ndz_, self.nz_
        alpha = self.alpha
        eta = self.eta
        nd = np.sum(ndz, axis=1).astype(np.intc)
        return llda._lda._loglikelihood(nzw, ndz, nz, nd, alpha, eta)

    def _sample_topics(self, rands):
        n_topics, vocab_size = self.nzw_.shape
        alpha = np.repeat(self.alpha, n_topics).astype(np.float64)
        eta = np.repeat(self.eta, vocab_size).astype(np.float64)
        llda._lda._sample_topics(self.WS, self.DS, self.ZS, self.nzw_, self.ndz_, self.nz_, alpha, eta, rands)

