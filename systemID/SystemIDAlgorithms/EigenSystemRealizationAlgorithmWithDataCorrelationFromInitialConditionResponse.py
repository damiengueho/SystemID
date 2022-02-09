"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""


import numpy as np
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power as matpow


from systemID.SystemIDAlgorithms.IdentificationInitialCondition import identificationInitialCondition
from systemID.ClassesGeneral.ClassSignal import DiscreteSignal
from systemID.SystemIDAlgorithms.GetMACandMSV import getMACandMSV




def eigenSystemRealizationAlgorithmWithDataCorrelationFromInitialConditionResponse(output_signals, state_dimension, input_dimension, **kwargs):
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

    # Hankel matrices
    H = np.zeros([p * output_dimension, q * number_signals, gamma + 1])
    for i in range(p):
        for j in range(q):
            for k in range(gamma + 1):
                H[i * output_dimension:(i + 1) * output_dimension, j * number_signals:(j + 1) * number_signals, k] = markov_parameters[i + j + 1 + k]

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
    X0 = np.matmul(LA.pinv(Op[0:p * output_dimension, :]), H[:, :, 0])[:, 0:number_signals]
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
