# -*- coding: utf-8 -*-
# @Date  : Sun Mar 18 20:24:37 2018
# @Author: Shaoze LUO
# @Notes : Affinity Propagation
import numpy as np


def ap(s, iters=100):
    a = np.zeros_like(s)
    r = np.zeros_like(s)
    rows = s.shape[0]
    for _ in range(iters):
        tmp_as = a + s
        max_tmp_as = np.tile(tmp_as.max(1), (rows, 1)).T
        max_tmp_as[range(rows), tmp_as.argmax(1)] = tmp_as[
            range(rows), tmp_as.argpartition(-2, 1)[:, -2]]
        r = s - max_tmp_as

        max_r = np.maximum(0, r)
        a = np.minimum(0, r.diagonal() + max_r.sum(0) -
                       max_r.diagonal() - max_r)
        a[range(rows), range(rows)] = max_r.sum(0) - max_r.diagonal()
    return a, r


def ap_raw(s, iters=100):
    a = np.zeros_like(s)
    r = np.zeros_like(s)
    rows = s.shape[0]
    for _ in range(iters):
        for i in range(rows):
            for k in range(rows):
                r[i, k] = s[i, k] - max([a[i, j] + s[i, j]
                                         for j in range(rows) if j != k])
        for i in range(rows):
            for k in range(rows):
                a[i, k] = min(0, r[k, k] + sum([max(0, r[j, k]) for j in range(rows) if (j != i) and (j != k)]))
            a[i, i] = sum([max(0, r[j, i]) for j in range(rows) if j != i])
    return a, r
