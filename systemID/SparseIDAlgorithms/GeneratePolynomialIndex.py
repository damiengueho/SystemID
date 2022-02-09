"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""


import numpy as np


def generatePolynomialIndex(dimension, order, post_treatment, **kwargs):
    a = np.ones([dimension])
    order = order + a

    index = np.zeros([int(order[0]), 1])
    for i in range(int(order[0])):
        index[i, 0] = i + 1

    for i in range(1, dimension):
        repel = index
        repsize = len(index[:, 0])
        repwith = np.ones([repsize, 1])

        for j in range(2, int(order[i]) + 1):
            repwith = np.concatenate((repwith, np.ones([repsize, 1]) * j), axis=0)

        index = np.concatenate((np.kron(np.ones((int(order[i]), 1)), repel), repwith), axis=1)

    s = index.shape
    o = np.ones([s[0], s[1]])
    index = index - o
    index = index.astype(int)

    if post_treatment:
        max_order = kwargs.get('max_order', order)
        rows = []
        for i in range(s[0]):
            if np.sum(index[i, :]) > max_order:
                rows.append(i)

        index = np.delete(index, rows, axis=0)

    return index
