import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.cluster import k_means


class MixGaussion(object):
    def __init__(self, n_componments=1, init_method='kmeans', max_iter=100):
        self.n_componments = n_componments
        self.init_method = init_method
        self.max_iter = max_iter

    def _initialize(self, X):
        n_componments = self.n_componments
        n_samples, n_features = X.shape
        weights = np.ones(n_componments) / n_componments
        if self.init_method == 'kmeans':
            means = k_means(X, n_clusters=n_componments)[0]
        else:
            means = X[np.random.choice(n_samples, n_componments, replace=False)]
        covs = [np.eye(n_features) for _ in range(n_componments)]

        return n_componments, n_samples, n_features, weights, means, covs

    def fit_predict(self, X, y=None):
        """
        Parameters
        ------
        X: (n_samples, n_features)
        weights: (n_componments, )
        means: (n_componments, n_features)
        covs: (n_componments, n_features, n_features)
        posterior: (n_samples, n_componments)
        pms: (n_samples, n_componments)
        """

        n_componments, n_samples, n_features, weights, means, covs = self._initialize(
            X)
        self.weights = weights
        self.means = means
        self.covs = covs

        for _ in range(self.max_iter):
            pms = np.stack([multivariate_normal.pdf(X, mean=means[k], cov=covs[k]) for k in range(n_componments)], axis=1)
            posterior = weights * pms
            posterior /= posterior.sum(1, keepdims=True)
            means = np.dot(posterior.T, X)
            means /= posterior.sum(0, keepdims=True).T
            covs = [np.dot(
                (X - means[k]).T, (X - means[k])*posterior[:, k][:, np.newaxis])/posterior[:, k].sum() for k in range(n_componments)]
            weights = posterior.sum(0) / n_samples

        res = posterior.argmax(1)

        return res

    def fit(self, X, y=None):
        self.fit_predict(X, y)
        return self

    def predict(self, X):
        pms = np.stack([multivariate_normal.pdf(X, mean=self.means[k], cov=self.covs[k])
                        for k in range(self.n_componments)], axis=1)
        posterior = self.weights * pms
        return posterior.argmax(1)


if __name__ == "__main__":
    X1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 10)
    X2 = np.random.multivariate_normal([2, 5], [[2, 1], [1, 2]], 10)
    X3 = np.random.multivariate_normal([11, 11], [[11, 1], [1, 11]], 10)
    X = np.concatenate([X1, X2, X3])

    mg = MixGaussion(3, 100)
    mg.fit(X)
    print('Total', mg.predict(X))
    print('X3', mg.predict(X3))

    gm = GaussianMixture(3)
    gm.fit(X)
    print('Total', gm.predict(X))
    print('X3', gm.predict(X3))

