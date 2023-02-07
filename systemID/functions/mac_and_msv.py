"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""


import numpy as np
import scipy.linalg as LA


def mac_and_msv(A_id, B_id, C_id, Rq, q):
    """
    Purpose:


    Parameters:
        -

    Returns:
        -

    Imports:
        -

    Description:


    See Also:
        -
    """

    # Shapes
    n, r = B_id.shape
    m, _ = C_id.shape

    # Initialize MAC and MSV
    MAC = np.zeros(n)
    MSV = np.zeros(n)

    # Transform to modal coordinates
    ev, T = LA.eig(A_id)
    A_m = np.diag(ev)
    B_m = np.matmul(LA.inv(T), B_id)
    C_m = np.matmul(C_id, T)
    Rq_m = np.matmul(LA.inv(T), Rq)

    # Controllability-like matrix
    Q = np.zeros(Rq.shape, dtype=complex)

    # Compute MAC and MSV
    for k in range(n):
        s_lambda = 0
        for i in range(q):
            Q[k, r*i:r*(i+1)] = A_m[k, k] ** i * B_m[k, :]
            s_lambda = s_lambda + np.abs(A_m[k, k]**i)
        MAC[k] = np.abs(np.dot(Q[k, :], np.conj(Rq_m[k, :]))) / np.sqrt(np.abs(np.dot(Q[k, :], np.conj(Q[k, :])))*np.abs(np.dot(Rq_m[k, :], np.conj(Rq_m[k, :]))))
        MSV[k] = np.sqrt(LA.norm(C_m[:, k]) * s_lambda * LA.norm(B_m[k, :]))

    return MAC, MSV
