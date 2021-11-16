"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


import numpy as np
from scipy import linalg as LA

from scipy.linalg import null_space


def qMarkovCover(markov_parameters, covariance_parameters, Q, state_dimension, **kwargs):

    # Dimensions
    (output_dimension, input_dimension) = markov_parameters[0].shape


    # Type and get WQ matrix
    type = kwargs.get('type', 'stochastic')
    if type == 'stochastic':
        covariance_input = kwargs.get('covariance_input', np.eye(input_dimension))
        WQ = np.kron(np.eye(Q), covariance_input)
    else:
        magnitude_channels_input = kwargs.get('magnitude_channels_input', np.eye(input_dimension))
        WQ = np.kron(np.eye(Q), magnitude_channels_input)


    ## Initialize HQ, RQ, MQ matrices
    HQ = np.zeros([output_dimension * Q, input_dimension * Q])
    RQ = np.zeros([output_dimension * Q, output_dimension * Q])
    MQ = np.zeros([output_dimension * (Q - 1), input_dimension])


    # Fill HQ, RQ, MQ matrices
    for i in range(Q):
        for j in range(i + 1):
            HQ[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension] = markov_parameters[i - j]
            RQ[i * output_dimension:(i + 1) * output_dimension, j * output_dimension:(j + 1) * output_dimension] = covariance_parameters[i - j]
        for j in range(i + 1, Q):
            RQ[i * output_dimension:(i + 1) * output_dimension, j * output_dimension:(j + 1) * output_dimension] = np.conj(covariance_parameters[j - i])
    for i in range(Q - 1):
        MQ[i * output_dimension:(i + 1) * output_dimension, :] = markov_parameters[i + 1]


    ## Compute DQ matrix
    DQ = RQ - np.matmul(np.matmul(HQ, WQ), np.transpose(HQ))


    ## SVD of DQ
    (R1, sigma1, St1) = LA.svd(DQ, full_matrices=True)
    Sigma1 = np.diag(sigma1)


    ## Truncate for the order of the system
    Rn1 = R1[:, 0:state_dimension]
    Snt1 = St1[0:state_dimension, :]
    Sigman1 = Sigma1[0:state_dimension, 0:state_dimension]


    ## Get PQn, P, Pb
    PQn = np.matmul(Rn1, LA.sqrtm(Sigman1))
    P = PQn[0:(Q-1) * output_dimension, :]
    Pb = PQn[output_dimension:Q * output_dimension, :]


    ## Identified matrices
    A_id = np.matmul(LA.pinv(P), Pb)
    B_id = np.matmul(LA.pinv(P), MQ)
    C_id = P[0:output_dimension, :]
    D_id = markov_parameters[0]

    return A_id, B_id, C_id, D_id


