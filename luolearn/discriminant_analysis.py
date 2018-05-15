import numpy as np
from .utils import check_X_y


class LinearDiscriminantAnalysis(object):
    def __init__(self, solver='svd', shrinkage=None, priors=None,
                 n_components=None, store_covariance=False, tol=1e-4):
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance
        self.tol = tol

    def _solve_lsqr(self, X, y, shrinkage):
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(X, y, self.priors_, shrinkage)
        self.coef_ = []
        self.intercept_ = []

    def _solve_svd(self, X, y):
        n_sapmles, n_features = X.shape
        n_classes = []

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        if self.priors is None:
            _, y_t = np.unique(y, return_counts=True)
            self.priors_ = y_t / float(len(y))
        else:
            self.priors_ = np.asarray(self.priors)
