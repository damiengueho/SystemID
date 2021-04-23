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


def eigenSystemRealizationAlgorithmFromInitialConditionResponse(output_signals, state_dimension, input_dimension, **kwargs):

    # Number of Signals
    number_signals = len(output_signals)

    # Number of steps
    number_steps = output_signals[0].number_steps

    # Dimensions
    output_dimension = output_signals[0].dimension

    # Building pseudo Markov parameters
    markov_parameters = []
    for i in range(number_steps):
        Yk = np.zeros([output_dimension, number_signals])
        for j in range(number_signals):
            Yk[:, j] = output_signals[j].data[:, i]
        markov_parameters.append(Yk)

    # Sizes
    p = kwargs.get('p', int(np.floor((len(markov_parameters) - 1) / 2)))
    if markov_parameters[0].shape == ():
        (output_dimension, number_signals) = (1, 1)
    else:
        (output_dimension, number_signals) = markov_parameters[0].shape

    # Hankel matrices H(0) and H(1)
    H0 = np.zeros([p * output_dimension, p * number_signals])
    H1 = np.zeros([p * output_dimension, p * number_signals])
    for i in range(p):
        for j in range(p):
            H0[i * output_dimension:(i + 1) * output_dimension, j * number_signals:(j + 1) * number_signals] = markov_parameters[i + j]
            H1[i * output_dimension:(i + 1) * output_dimension, j * number_signals:(j + 1) * number_signals] = markov_parameters[i + j + 1]

    # SVD H(0)
    (R, sigma, St) = LA.svd(H0, full_matrices=True)
    Sigma = np.diag(sigma)

    # Matrices Rn, Sn, Sigman
    Rn = R[:, 0:state_dimension]
    Snt = St[0:state_dimension, :]
    Sigman = Sigma[0:state_dimension, 0:state_dimension]

    # Identified matrices
    Op = np.matmul(Rn, matpow(Sigman, 1 / 2))
    Rq = np.matmul(matpow(Sigman, 1 / 2), Snt)
    A_id = np.matmul(LA.pinv(Op), np.matmul(H1, LA.pinv(Rq)))
    x0 = Rq[:, 0:number_signals]
    C_id = Op[0:output_dimension, :]


    def A(tk):
        return A_id

    def B(tk):
        return np.zeros([state_dimension, input_dimension])

    def C(tk):
        return C_id

    def D(tk):
        return(np.zeros([output_dimension, input_dimension]))


    return A, B, C, D, x0, H0, H1, R, Sigma, St, Rn, Sigman, Snt, Op, Rq
