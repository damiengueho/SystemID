"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


import numpy as np


def getTimeVaryingMarkovParameters(A, B, C, D, tk, dt, number_steps):

    if number_steps < 0:
        return np.zeros(D(tk).shape)
    elif number_steps == 0:
        return D(tk)
    elif number_steps == 1:
        return np.matmul(C(tk + dt), B(tk))
    else:
        Phi = B(tk)
        for i in range(1, number_steps):
            Phi = np.matmul(A(tk + i*dt), Phi)
        return np.matmul(C(tk + number_steps*dt), Phi)


def getTimeVaryingMarkovParameters_matrix(A, B, C, D, k, number_steps):

    if number_steps < 0:
        return np.zeros(D[:, :, k].shape)
    elif number_steps == 0:
        return D[:, :, k]
    elif number_steps == 1:
        return np.matmul(C[:, :, k + 1], B[:, :, k])
    else:
        Phi = B[:, :, k]
        for i in range(1, number_steps):
            Phi = np.matmul(A[:, :, k + i], Phi)
        return np.matmul(C[:, :, k + number_steps], Phi)
