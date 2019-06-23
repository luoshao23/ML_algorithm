import numpy as np


class Percepton(object):

    def __init__(self, eta=0.1):
        self._eta = eta

    def fit(self, X, y, batch=1):
        dim = X.shape[1]
        # W = np.random.randn(dim)
        # b = np.random.randn()
        W = np.zeros(dim)
        b = 0
        self._W = W
        self._b = b

        while True:
            mul = y * (np.dot(X, W) + b)
            ind = np.argwhere(mul <= 0).ravel()
            if len(ind) > 0:
                i = np.random.choice(ind, size=batch, replace=True)
                W += self._eta * (X[i, :] * y[i]).sum() / batch
                b += self._eta * y[i].sum() / batch
            else:
                break

        self._W = W
        self._b = b

        return self

    def predict(self, X, y=None):
        if hasattr(self, '_W'):
            res = np.dot(X, self._W) + self._b
            res = np.sign(res)
        else:
            raise RuntimeError('Please fit first.')
        return res


if __name__ == "__main__":
    X = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])

    clf = Percepton(0.1)
    clf.fit(X, y)
    print(clf._W, clf._b)
    print(clf.predict(X))
