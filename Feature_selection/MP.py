import numpy as np
from copy import deepcopy


def MP(ft, D, tol=1e-4):
    """
    This is an implementation of matching pursuit (MP) algorithm
    Input: Signal: f(t), dictionary D with normalized columns g_i
    Output: List if coefficients a_n and indices for corresponding atoms gamma_n
    Initialization:
        R1 = f(t)
        n = 1
    Repeat:
        Find g_{gamma_n} with maximum inner product <R_n, g_{gamma_n}>
        a_n = <R_n, g_{gamma_n}>
        R_{n + 1} = R_n - a_n * g_{gamma_n}
        n = n + 1
    Util stop condition (e.g., ||R_n|| < threshold)
    return
    """
    R = deepcopy(ft)
    a = []
    gamma = []
    D = D / np.linalg.norm(D, axis=0)

    while np.dot(R, R) > tol:
        gammai = np.argmax(np.abs(np.dot(R, D)))
        ai = np.dot(R, D[:, gammai])
        R -= ai * D[:, gammai]
        a.append(ai)
        gamma.append(gammai)

    a = np.asarray(a)
    gamma = np.asarray(gamma)

    return a, gamma


if __name__ == "__main__":
    v = np.random.random(8)
    print('The original: ', v)
    D = np.random.random((8, 12))
    a, gamma = MP(v, D)

    # print(a, gamma)
    D /= np.linalg.norm(D, axis=0)
    vn = np.sum(a * D[:, gamma], axis=1)
    print('The approximation: ', vn)


