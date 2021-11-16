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

from SystemIDAlgorithms.IdentificationInitialCondition import identificationInitialCondition
from ClassesGeneral.ClassSignal import DiscreteSignal
from SystemIDAlgorithms.GetMACandMSV import getMACandMSV


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
    min_size = int(np.floor((len(markov_parameters) - 1) / 2))
    p = kwargs.get('p', min_size)
    p = min(p, min_size)
    print('p in ERA =', p)
    q = kwargs.get('q', p)
    q = min(q, min_size)
    print('q in ERA =', q)
    print('min_size =', min_size)
    if markov_parameters[0].shape == ():
        (output_dimension, number_signals) = (1, 1)
    else:
        (output_dimension, number_signals) = markov_parameters[0].shape

    # Hankel matrices H(0) and H(1)
    H0 = np.zeros([p * output_dimension, q * number_signals])
    H1 = np.zeros([p * output_dimension, q * number_signals])
    for i in range(p):
        for j in range(q):
            H0[i * output_dimension:(i + 1) * output_dimension, j * number_signals:(j + 1) * number_signals] = markov_parameters[i + j]
            H1[i * output_dimension:(i + 1) * output_dimension, j * number_signals:(j + 1) * number_signals] = markov_parameters[i + j + 1]

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
    Op = np.matmul(Rn, matpow(Sigman, 1 / 2))
    Rq = np.matmul(matpow(Sigman, 1 / 2), Snt)
    A_id = np.matmul(LA.pinv(Op), np.matmul(H1, LA.pinv(Rq)))
    X0 = Rq[:, 0:number_signals]
    C_id = Op[0:output_dimension, :]


    def A(tk):
        return A_id

    def B(tk):
        return np.zeros([state_dimension, input_dimension])

    def C(tk):
        return C_id

    def D(tk):
        return np.zeros([output_dimension, input_dimension])


    # x0 = identificationInitialCondition(DiscreteSignal(input_dimension, true_output_signal.total_time, true_output_signal.frequency), true_output_signal, A, B, C, D, 0, p)


    return A, B, C, D, X0, H0, H1, R, Sigma, St, Rn, Sigman, Snt, Op, Rq, MAC, MSV
