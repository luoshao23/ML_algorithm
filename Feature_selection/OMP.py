import numpy as np
from copy import deepcopy
import pandas as pd


def OMP(ft, D, tol=1e-4, K=20):
    R = deepcopy(ft)
    gamma = []
    D = D / np.linalg.norm(D, axis=0)
    t = 1
    while t < K:
        gammai = np.argmax(np.abs(np.dot(R, D)))
        gamma.append(gammai)
        fi = D[:, gamma]
        x = np.dot(np.linalg.pinv(fi), ft)
        R = ft - np.dot(fi, x)
        if np.dot(R, R) < tol:
            break
        t += 1

    x = np.asarray(x)
    gamma = np.asarray(gamma)

    return x, gamma


if __name__ == "__main__":
    v = np.random.random(8)
    print('The original: ', v)
    # D = np.random.random((8, 12))
    D = np.random.randn(8, 12)
    a, gamma = OMP(v, D)
    D /= np.linalg.norm(D, axis=0)
    vn = np.sum(a * D[:, gamma], axis=1)
    print('The approximation: ', vn)
