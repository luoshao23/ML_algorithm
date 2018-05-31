import warnings

import numpy as np
from scipy import linalg
from sklearn.preprocessing import StandardScaler
from .covariance import ledoit_wolf, empirical_covariance


from .externals.six import string_types
from .utils import check_X_y

# from .preprocessing import StandardScaler


def _cov(X, shrinkage=None):
    shrinkage = 'emprical' if shrinkage is None else shrinkage
    if isinstance(shrinkage, string_types):
        if shrinkage == 'auto':
            sc = StandardScaler()
            X = sc.fit_transform(X)
            s = ledoit_wolf(X)[0]
            s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
        elif shrinkage == 'empirical':
            s = empirical_covariance(X)
        else:
            raise ValueError('unknown shrinkage parameter')


def _class_means(X, y):
    means = []
    classes = np.unique(y)
    for g in classes:
        Xg = X[y == g, :]
        means.append(Xg.mean(0))
    return np.asarray(means)


def _class_cov(X, y, priors=None, shrinkage=None):
    covs = []
    classes = np.unique(y)
    for g in classes:
        Xg = X[y == g, :]
        covs.append(np.atleast_2d(_cov(Xg, shrinkage)))
    return np.average(covs, axis=0, weights=priors)


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
        self.coef_ = linalg.lstsq(self.covariance_, self.means_.T)[0].T
        self.intercept_ = (-0.5 * np.diag(np.dot(
            self.means_, self.coef_.T)) + np.log(self.priors_))

    def _solve_svd(self, X, y):
        n_sapmles, n_features = X.shape
        n_classes = len(self.classes_)
        self.means_ = _class_means(X, y)

        if self.store_covariance:
            self.covariance_ = _class_cov(X, y, self.priors_)

        Xc = []
        for idx, group in enumerate(self.classes_):
            Xg = X[y == group, :]
            Xc.append(Xg - self.means_[idx])

        self.xbar_ = np.dot(self.priors_, self.means_)
        Xc = np.concatenate(Xc, axis=0)

        std = Xc.std(axis=0)

        std[std == 0] = 1.
        fac = 1. / (n_sapmles - n_classes)

        X = np.sqrt(fac) * (Xc / std)

        U, S, V = linalg.svd(X, full_matrices=False)

        rank = np.sum(S > self.tol)
        if rank < n_features:
            warnings.warn("Variable are collinear.")

        scalings = (V[:rank] / std).T / S[:rank]

        X = np.dot(((np.sqrt((n_samples * self.priors_) * fac)) *
                    (self.means_ - self.xbar_).T).T, scalings)

        _, S, V = linalg.svd(X, full_matrices=0)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        if self.priors is None:
            _, y_t = np.unique(y, return_counts=True)
            self.priors_ = y_t / float(len(y))
        else:
            self.priors_ = np.asarray(self.priors)

        if (self.priors_ < 0).any():
            raise ValueError('priors must be non-negative')
        if self.priors_.sum() != 1:
            warnings.warn('priors not sum to 1. Renormalizing')
            self.priors_ = self.priors_ / self.priors_.sum()

        if self.n_components is None:
            self._max_components = len(self.classes_) - 1
        else:
            self._max_components = min(len(self.classes_) - 1, self.n_components)

        if self.solver == 'svd':
            if self.shrinkage is not None:
                raise NotImplementedError('shrinkage not supported')
            self._solve_svd(X, y)
