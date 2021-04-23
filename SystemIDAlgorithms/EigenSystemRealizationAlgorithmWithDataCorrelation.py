"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 10
Date: April 2021
Python: 3.7.7
"""


import numpy as np
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power as matpow


def eigenSystemRealizationAlgorithmWithDataCorrelation(markov_parameters, tau, state_dimension):

    # Sizes
    p = int(np.floor((len(markov_parameters)-1)/(2*(1+tau))))
    xi = p
    gamma = 1 + 2 * xi * tau
    if markov_parameters[0].shape == ():
        (output_dimension, input_dimension) = (1, 1)
    else:
        (output_dimension, input_dimension) = markov_parameters[0].shape

    # Hankel matrices
    H = np.zeros([p * output_dimension, p * input_dimension, gamma + 1])
    for i in range(p):
        for j in range(p):
            for k in range(gamma + 1):
                H[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension, k] = markov_parameters[i + j + 1 + k]

    # Data Correlation Matrices
    R = np.zeros([p * output_dimension, p * output_dimension, gamma + 1])
    for i in range(gamma + 1):
        R[:, :, i] = np.matmul(H[:, :, i], np.transpose(H[:, :, 0]))

    # Building Block Correlation Hankel Matrices
    H0 = np.zeros([(xi + 1) * p * output_dimension, (xi + 1) * p * output_dimension])
    H1 = np.zeros([(xi + 1) * p * output_dimension, (xi + 1) * p * output_dimension])
    for i in range(xi + 1):
        for j in range(xi + 1):
            H0[i * p * output_dimension:(i + 1) * p * output_dimension, j * p * output_dimension:(j + 1) * p * output_dimension] = R[:, :, (i + j) * tau]
            H1[i * p * output_dimension:(i + 1) * p * output_dimension, j * p * output_dimension:(j + 1) * p * output_dimension] = R[:, :, (i + j) * tau + 1]

    # SVD H(0)
    print(H0.shape)
    (R, sigma, St) = LA.svd(H0, full_matrices=True)
    Sigma = np.diag(sigma)

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

    return A, B, C, D, H0, H1, R, Sigma, St, Rn, Sigman, Snt, Op, Rq
