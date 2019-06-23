import heapq
import numpy as np
import matplotlib.pyplot as plt


class KDTree(object):

    class _Node:
        def __init__(self, val=None, left=None, right=None, parent=None, axis=None, idx=None):
            self._val = val
            self._left = left
            self._right = right
            self._parent = parent
            self._axis = axis
            self._idx = idx

        def __str__(self):
            return '%s(%s, %s)' % (self._val, self._left, self._right)

        __repr__ = __str__

        @property
        def sibling(self):
            if self._parent:
                if self._parent._left and self._parent._left is self:
                    return self._parent._right
                elif self._parent._right and self._parent._right is self:
                    return self._parent._left

        def is_leaf(self):
            return self._left is None and self._right is None

    def ___init__(self):
        self._root = None

    def fit(self, X):
        self._k = X.shape[1]
        self._root = self._fit(X)

        return self

    def _fit(self, X, parent=None, depth=0):
        n = X.shape[0]
        if n == 0:
            return None
        j = depth % self._k
        mid = n // 2
        midind = np.argpartition(X[:, j], mid)

        c = self._Node(X[midind[mid]], None, None, parent, j, midind[mid])
        left = self._fit(X[midind[:mid]], c, depth+1)
        right = self._fit(X[midind[mid+1:]], c, depth+1)
        c._left = left
        c._right = right

        return c

    def nearest_neighbor(self, p, k=1):
        root = self._root
        results = []
        self._nearest_neighbor(root, k, results, p)
        return [(np.sqrt(-d), node) for d, node in sorted(results)]

    def _nearest_neighbor_depreciated(self, node, k, results, p, seen=set()):
        # if node is root, stop
        if node is None:
            return

        # forward search for the leaf as the current closest point
        while not node.is_leaf():
            if p[node._axis] < node._val[node._axis]:
                if node._left:
                    node = node._left
                else:
                    break
            else:
                if node._right:
                    node = node._right
                else:
                    break

        r2 = self._distance(node._val, p)

        if len(results) >= k:
            if - results[0][0] > r2:
                heapq.heapreplace(results, (-r2, node._val))
        else:
            heapq.heappush(results, (-r2, node._val))

        # backward recursively
        while node and node._idx not in seen:
            seen.add(node._idx)
            parent = node._parent
            if parent and (parent._val[parent._axis] - p[parent._axis])**2 < -results[0][0]:
                self._nearest_neighbor(node.sibling, k, results, p, seen)
            node = node._parent

    def _nearest_neighbor(self, node, k, results, p):
        if not node:
            return

        r2 = self._distance(node._val, p)

        if len(results) >= k:
            if - results[0][0] > r2:
                heapq.heapreplace(results, (-r2, node._val))
        else:
            heapq.heappush(results, (-r2, node._val))

        if p[node._axis] < node._val[node._axis]:
            if node._left is not None:
                self._nearest_neighbor(node._left, k, results, p)
        else:
            if node._right is not None:
                self._nearest_neighbor(node._left, k, results, p)

        if (node._val[node._axis] - p[node._axis])**2 < -results[0][0]:
            if p[node._axis] < node._val[node._axis]:
                if node._right is not None:
                    self._nearest_neighbor(node._left, k, results, p)
            else:
                if node._left is not None:
                    self._nearest_neighbor(node._left, k, results, p)





    def _distance(self, p1, p2):
        return sum([(p1[i] - p2[i])**2 for i in range(self._k)])

    def display(self):
        root = self._root
        fig, ax = plt.subplots(1, 1)
        self._ax = ax
        self._display(root)
        plt.show()
        return fig

    def _display(self, node):
        if node is None:
            return
        ax = self._ax
        xh, xl, yh, yl = 10, 0, 10, 0
        x0, y0 = node._val
        axis = node._axis
        if axis == 1:
            yh = yl = y0

            if node._parent:
                xt, yt = node._parent._val
                if xt > x0:
                    xh = xt
                else:
                    xl = xt
        else:
            xh = xl = x0

            if node._parent:
                xt, yt = node._parent._val
                if yt > y0:
                    yh = yt
                else:
                    yl = yt

        ax.plot(x0, y0, 'k.')
        ax.plot([xl, xh], [yl, yh], 'k-')
        self._display(node._left)
        self._display(node._right)


if __name__ == "__main__":
    X = np.array([[2, 3],
                  [5, 4],
                  [9, 6],
                  [4, 7],
                  [8, 1],
                  [7, 2]])

    kd = KDTree()
    kd.fit(X)
    # print(kd.nearest_neighbor([2.1, 3.1]))
    print(kd.nearest_neighbor([2, 4.5]))
    # print(kd._root)
    # kd.display()
