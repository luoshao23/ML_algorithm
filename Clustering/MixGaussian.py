import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


class MixGaussion(object):
    def __init__(self, n_componments=1, max_iter=100):
        self.n_componments = n_componments
        self.max_iter = max_iter

    def fit(self, X, y=None):
        '''
        X: (n_samples, n_features)
        weights: (n_componments, )
        means: (n_componments, n_features)
        covs: (n_componments, n_features, n_features)
        posterior: (n_samples, n_componments)
        pms: (n_samples, n_componments)
        '''
        n_componments = self.n_componments
        m, d = X.shape
        weights = np.ones(n_componments) / n_componments
        means = X[np.random.choice(m, n_componments, replace=False)]
        covs = [0.1 * np.eye(d) for _ in range(n_componments)]

        for _ in range(self.max_iter):
            pms = np.stack([multivariate_normal.pdf(X, mean=means[k], cov=covs[k]) for k in range(n_componments)], axis=1)
            posterior = weights * pms
            posterior /= posterior.sum(1, keepdims=True)
            means = np.dot(posterior.T, X)
            means /= posterior.sum(0, keepdims=True).T
            covs = [np.dot(
                (X - means[k]).T, (X - means[k])*posterior[:, k][:, np.newaxis])/posterior[:, k].sum() for k in range(n_componments)]
            weights = posterior.sum(0) / m

        res = posterior.argmax(1)

        return res


if __name__ == "__main__":
    X1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 10)
    X2 = np.random.multivariate_normal([2, 5], [[2, 1], [1, 2]], 10)
    X3 = np.random.multivariate_normal([11, 11], [[11, 1], [1, 11]], 10)
    X = np.concatenate([X1, X2, X3])

    mg = MixGaussion(3, 300)
    res = mg.fit(X)
    print(res)

    gm = GaussianMixture(3)
    gm.fit(X)
    res = gm.predict(X2)
    print(res)




