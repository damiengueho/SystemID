"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


import numpy as np
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power as matpow

from SystemIDAlgorithms.GetMACandMSV import getMACandMSV




def eigenSystemRealizationAlgorithmWithDataCorrelation(markov_parameters, state_dimension, **kwargs):

    # Sizes
    min_size = int(np.floor(np.sqrt((len(markov_parameters) - 1) / 4)))
    p = int(kwargs.get('p', min_size))
    p = min(p, min_size)
    q = int(kwargs.get('q', p))
    q = min(q, min_size)
    xi = int(kwargs.get('xi', p))
    xi = min(xi, min_size)
    zeta = int(kwargs.get('zeta', p))
    zeta = min(zeta, min_size)
    tau = int(kwargs.get('tau', p))
    tau = min(tau, min_size)
    gamma = 1 + (xi + zeta) * tau

    # Dimensions
    (output_dimension, input_dimension) = markov_parameters[0].shape

    # Hankel matrices
    H = np.zeros([p * output_dimension, q * input_dimension, gamma + 1])
    for i in range(p):
        for j in range(q):
            for k in range(gamma + 1):
                H[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension, k] = markov_parameters[i + j + 1 + k]

    # Data Correlation Matrices
    R = np.zeros([p * output_dimension, p * output_dimension, gamma + 1])
    for i in range(gamma + 1):
        R[:, :, i] = np.matmul(H[:, :, i], np.transpose(H[:, :, 0]))

    # Building Block Correlation Hankel Matrices
    H0 = np.zeros([(xi + 1) * p * output_dimension, (zeta + 1) * p * output_dimension])
    H1 = np.zeros([(xi + 1) * p * output_dimension, (zeta + 1) * p * output_dimension])
    for i in range(xi + 1):
        for j in range(zeta + 1):
            H0[i * p * output_dimension:(i + 1) * p * output_dimension, j * p * output_dimension:(j + 1) * p * output_dimension] = R[:, :, (i + j) * tau]
            H1[i * p * output_dimension:(i + 1) * p * output_dimension, j * p * output_dimension:(j + 1) * p * output_dimension] = R[:, :, (i + j) * tau + 1]

    # SVD H(0)
    (R, sigma, St) = LA.svd(H0, full_matrices=True)
    Sigma = np.diag(sigma)

    # MAC and MSV
    mac_and_msv = kwargs.get('mac_and_msv', False)
    if mac_and_msv:
        pm, qr = H0.shape
        n = min(pm, qr)
        Rn = R[:, 0:n]
        Snt = St[0:n, :]
        Sigman = Sigma[0:n, 0:n]
        Op = np.matmul(Rn, LA.sqrtm(Sigman))
        Rq = np.matmul(LA.sqrtm(Sigman), Snt)
        A_id = np.matmul(LA.pinv(Op), np.matmul(H1, LA.pinv(Rq)))
        B_id = Rq[:, 0:input_dimension]
        C_id = Op[0:output_dimension, :]
        MAC, MSV = getMACandMSV(A_id, B_id, C_id, Rq, p)
    else:
        MAC = []
        MSV = []

    # Matrices Rn, Sn, Sigman
    Rn = R[:, 0:state_dimension]
    Snt = St[0:state_dimension, :]
    Sigman = Sigma[0:state_dimension, 0:state_dimension]

    # Identified matrices
    Op = np.matmul(Rn, matpow(Sigman, 1/2))
    Rq = np.matmul(matpow(Sigman, 1/2), Snt)
    A_id = np.matmul(LA.pinv(Op), np.matmul(H1, LA.pinv(Rq)))
    B_id = np.matmul(LA.pinv(Op[0:p * output_dimension, :]), H[:, :, 0])[:, 0:input_dimension]
    C_id = Op[0:output_dimension, :]
    D_id = markov_parameters[0]

    def A(tk):
        return A_id

    def B(tk):
        return B_id

    def C(tk):
        return C_id

    def D(tk):
        return D_id

    return A, B, C, D, H0, H1, R, Sigma, St, Rn, Sigman, Snt, Op, Rq, MAC, MSV
