import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier as AC
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split


class AdaBoostClassifier(object):

    def __init__(self, base_estimator=None, n_estimators=50):
        self.base = base_estimator if base_estimator else DecisionTreeClassifier(max_depth=1)
        self.n_estimators = n_estimators
        self.estimators = []
        self.estimator_weights = []

    def fit(self, X, y, weights=None):
        m, _ = X.shape
        weights = np.asarray(weights) if weights else np.ones(m) / m
        estimator = self.base
        self.errors = []

        for _ in range(self.n_estimators):
            clf = estimator.__class__(random_state=np.random.RandomState())
            clf.fit(X, y, sample_weight=weights)
            yhat = clf.predict(X)
            err = np.average(y != yhat, weights=weights)
            self.errors.append(err)
            if err <= 0:
                self.estimator_weights.append(1)
                self.estimators.append(clf)
                break
            if err > 0.5:
                break
            at = 0.5 * np.log((1 - err) / err)
            self.estimator_weights.append(at)
            self.estimators.append(clf)
            weights *= np.exp(-at*y*yhat)
            weights /= np.sum(weights)

        return self

    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for at, estimator in zip(self.estimator_weights, self.estimators):
            pred += at * estimator.predict(X)
        pred /= np.sum(self.estimator_weights)
        pred = np.sign(pred)

        return pred

    def score(self, X, y):
        yhat = self.predict(X)
        return np.sum(yhat == y) / len(y)


if __name__ == "__main__":
    # Construct dataset
    X1, y1 = make_gaussian_quantiles(cov=2.,
                                    n_samples=200, n_features=5,
                                    n_classes=2, random_state=1234)
    X2, y2 = make_gaussian_quantiles(mean=(3, 3, 3, 3, 3), cov=1.5,
                                    n_samples=300, n_features=5,
                                    n_classes=2, random_state=1234)

    X = np.concatenate((X1, X2))
    y = np.concatenate((2 * y1 - 1, 1 - 2 * y2))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ada = AdaBoostClassifier()
    ada.fit(X_train, y_train)
    print('My AdaBoost', ada.score(X_test, y_test))

    ac = AC(algorithm="SAMME")
    ac.fit(X_train, y_train)
    print('Sklearn\'s AdaBoost', ac.score(X_test, y_test))
