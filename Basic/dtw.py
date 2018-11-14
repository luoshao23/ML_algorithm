import numpy as np


def dist(a, b):
    return np.sqrt((a - b)**2)


def dtw_dist(s, t):
    n = len(s)
    m = len(t)
    dtw = np.zeros((n+1, m+1))

    dtw[:, 0] = np.Inf
    dtw[0, :] = np.Inf
    dtw[0, 0] = 0

    for i in range(n):
        for j in range(m):
            cost = dist(s[i], t[j])
            dtw[i+1, j+1] = cost + min(dtw[i, j+1], dtw[i+1, j], dtw[i, j])

    return dtw[-1, -1]


def test():
    a = [5, 2, 4, 2, 5]
    b = [1, 4, 2, 3]

    r = dtw_dist(a, b)
    print(r)

test()