import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


class cDBSCAN(object):
    def __init__(self, min_pts=5, epsilon=0.5, metric='euclidean'):
        self.min_pts = min_pts
        self.epsilon = epsilon
        self.metric = metric

    def fit(self, X, y=None):
        self.fit_predict(X, y)
        return self

    def predict(self, X):
        pass

    def fit_predict(self, X, y=None):
        n_samples, _ = X.shape
        nearin = cdist(X, X, metric=self.metric) <= self.epsilon
        near_num = np.sum(nearin, axis=1)
        core_ind = set(np.arange(n_samples)[near_num >= self.min_pts])
        print(core_ind)

        n_clusters = 0
        this_set = set(range(n_samples))
        clusters = []

        while core_ind:
            old_set = this_set.copy()
            ele = core_ind.pop()
            queue = [ele]
            this_set.remove(ele)
            while queue:
                q = queue.pop(0)
                if near_num[q] >= self.min_pts:
                    dlt = this_set.intersection(np.arange(n_samples)[nearin[q]])
                    queue.extend(dlt)
                    this_set.difference_update(dlt)
            n_clusters += 1
            C = old_set.difference(this_set)
            clusters.append(C)
            core_ind.difference_update(C)
        labels = -1 * np.ones(n_samples, dtype=int)

        for l, g in enumerate(clusters):
            labels[list(g)] = l
        self.labels = labels
        return labels

if __name__ == "__main__":
    np.random.seed(23)
    X = np.random.random((40, 2))
    cdb = cDBSCAN(epsilon=0.2)
    res = cdb.fit_predict(X)
    print(res)
    db = DBSCAN(eps=0.2)
    l = db.fit_predict(X)
    print(l)

    fig, (a1, a2) = plt.subplots(1, 2)

    a1.scatter(X[:, 0], X[:, 1], c=res)
    a2.scatter(X[:, 0], X[:, 1], c=l)
    plt.show()
