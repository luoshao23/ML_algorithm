from abc import abstractclassmethod
from collections import defaultdict
import warnings

import numpy as np
from scipy.special import logsumexp
from sklearn.utils import check_X_y
from sklearn.preprocessing import LabelBinarizer


class Naive_Bayes(object):
    """Naive Bayes classifier
    The input and the target is no need to binarized.

    """

    def __init__(self, alpha=0):
        self._unique_labels = None
        self._alpha = alpha

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._n_feature = X.shape[1]
        self._unique_labels, self._class_counts = np.unique(
            y, return_counts=True)
        self._class_prior = (self._class_counts + self._alpha) / (self._class_counts.sum() + len(self._unique_labels))
        self._count(X, y)

    def _count(self, X, y):
        self._matrix = [defaultdict(int) for _ in range(self._n_feature)]
        for j in range(self._n_feature):
            labels, counts = np.unique(
                np.c_[X[:, j], y], return_counts=True, axis=0)
            s = len(set(X[:, j]))
            for l, c in zip(labels, counts):
                self._matrix[j][tuple(l)] = (c + self._alpha) / (self._class_counts[self._unique_labels == l[1]][0] + s)

    def predict(self, X, y=None):
        n_samples, n_features = X.shape
        assert n_features == self._n_feature
        res = np.empty(n_samples)
        for i in range(n_samples):
            proba = self._class_prior
            for j in range(n_features):
                for c in range(len(self._unique_labels)):
                    proba[c] *= self._matrix[j][(X[i, j],
                                                 self._unique_labels[c])]
            res[i] = self._unique_labels[np.argmax(proba)]

        return res


class BaseNB(object):

    @abstractclassmethod
    def _joint_log_likelihood(self, X):
        """log P(c) + log P(X|c)
        """
    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        return jll - logsumexp(jll, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

_ALPHA_MIN = 1e-10


class BaseDiscreteNB(BaseNB):

    def _update_class_log_prior(self, class_prior=None):
        n_classes = len(self.classes_)
        if class_prior is not None:
            if len(class_prior) != n_classes:
                raise ValueError('Number dismatch.')
            self.class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            log_class_count = np.log(self.class_count_)
            self.class_log_prior_ = log_class_count - np.log(self.class_count_.sum())
        else:
            self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))

    def _check_alpha(self):
        if np.min(self.alpha) < 0:
            raise ValueError('Alpha should be > 0.')
        if isinstance(self.alpha, np.ndarray):
            if not self.alpha.shape[0] == self.feature_count_.shape[1]:
                raise ValueError("alpha should be a scalar or an array with shape [n_features]")
        if np.min(self.alpha) < _ALPHA_MIN:
            warnings.warn("alpha too samll, setting alpha = %.1e" % _ALPHA_MIN)
            return np.maximum(self.alpha, _ALPHA_MIN)
        return self.alpha

    def fit(self, X, y):
        X, y = check_X_y(X, y, 'csr')
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_

        # in case of binary label, turn it into shape [n_sample, 2]
        if Y.shape[1] == 1:
            Y = np.concatenate([1 - Y, Y], axis=1)

        class_prior = self.class_prior
        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.feature_count_ = np.zeros(
            (n_effective_classes, n_features), dtype=np.float64)
        self._count(X, Y)
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)


class MultinomialNB(BaseDiscreteNB):

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def _count(self, X, Y):
        # self.feature_count with shape [n_classes, n_features]
        # self.class_count_ with shape [n_classes]
        if np.any(X < 0):
            raise ValueError("Input X must be non-negative")
        self.feature_count_ += np.dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.feature_log_prob = (np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1)))

    def _joint_log_likelihood(self, X):
        return np.dot(X, self.feature_log_prob.T) + self.class_log_prior_


if __name__ == "__main__":
    # x1 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    # x2 = [3, 5, 5, 3, 3, 3, 5, 5, 7, 7, 7, 5, 5, 7, 7]
    # y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    x1 = [3, 2, 5]
    x2 = [4, 1, 8]
    y = [-1, 1, -1]
    X = np.vstack([x1, x2]).T
    # print(X)
    nb = Naive_Bayes(1)
    nb.fit(X, y)
    print(nb.predict(np.array([[2, 3], [3, 5]])))
    mnb = MultinomialNB()
    mnb.fit(X, y)
    print(mnb.predict(np.array([[8, 3], [3, 5]])))
