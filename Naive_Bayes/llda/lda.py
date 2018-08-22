import numpy as np

import llda.utils


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
        pass

    def _fit(self, X):
        random_state = llda.utils.check_random_state(self.random_state)
        rands = self._rands.copy()
        self._initialize(X)
        for it in range(self.n_iter):
            # FIXME: using numpy.roll with a random shift might be faster
            random_state.shuffle(rands)
            if it % self.refresh == 0:
                ll = self.loglikelihood()
                self.loglikelihoods_.append(ll)


    def _initialize(self, X):
        D, W = X.shape
        N = int(X.sum())
        n_topics = self.n_topics
        n_iter = self.n_iter

        self.nzw_ = nzw_ = np.zeros((n_topics, W), dtype=np.intc)
        self.ndz_ = ndz_ = np.zeros((D, n_topics), dtype=np.intc)
        self.nz_ = nz_ = np.zeros(n_topics, dtype=np.intc)

        self.WS, self.DS = WS, DS = llda.utils.matrix_to_lists(X)
        np.testing.assert_equal(N, len(WS))
        self.ZS = ZS = np.empty_like(self.WS, dtype=np.intc)
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

