"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""



import numpy as np
import scipy.linalg as LA

from systemID.SystemIDAlgorithms.IdentificationInitialCondition import identificationInitialCondition


def timeVaryingEigenSystemRealizationAlgorithmWithDataCorrelationFromInitialConditionResponse(free_decay_experiments, full_experiment, state_dimension, **kwargs):
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

    # Dimensions and number of steps
    input_dimension = free_decay_experiments[0].input_dimension
    output_dimension = free_decay_experiments[0].output_dimension
    number_free_decay_experiments = free_decay_experiments[0].number_experiments
    number_steps = free_decay_experiments[0].output_signals[0].number_steps

    # Sizes
    p = int(kwargs.get('p', 100))
    p = min(p, 100)
    xi = int(kwargs.get('xi', 100))
    xi = min(xi, 100)
    number_batches = len(free_decay_experiments)
    tau = int(kwargs.get('tau', 100))
    tau = min(tau, 100)
    max_k = number_steps - p - (xi - 1) * tau - 2

    # Compute free response experiments Y matrices
    Y = np.zeros([(p + 1) * output_dimension, number_free_decay_experiments, max_k + 2 + (xi - 1) * tau, number_batches])
    for l in range(number_batches):
        free_decay_outputs = free_decay_experiments[l].output_signals
        for k in range(max_k + 2 + (xi - 1) * tau):
            for i in range(p + 1):
                for j in range(number_free_decay_experiments):
                    Y[i * output_dimension:(i + 1) * output_dimension, j, k, l] = free_decay_outputs[j].data[:, i + k]


    # Frequency
    frequency = full_experiment.input_signals[0].frequency


    # Initializing Identified matrices
    A_id = np.zeros([state_dimension, state_dimension, number_steps])
    B_id = np.zeros([state_dimension, input_dimension, number_steps])
    C_id = np.zeros([output_dimension, state_dimension, number_steps])
    D_id = np.zeros([output_dimension, input_dimension, number_steps])

    # Store Generalized data correlation matrices
    Rkt = np.zeros([(p + 1) * output_dimension, (p + 1) * output_dimension, max_k + 2 + (xi - 1) * tau, number_batches])


    # Store observability matrices at each step (for eigenvalue check)
    Ok = np.zeros([xi * p * output_dimension, state_dimension, max_k + 1])
    Ok1 = np.zeros([xi * p * output_dimension, state_dimension, max_k + 1])


    # Store SVD at each time step
    sigma = []


    # Calculating first q A, B, C - Free Response

    # Build Generalized data correlation matrices
    Hpnt = np.zeros([(p + 1) * output_dimension, number_free_decay_experiments, max_k + 2 + (xi - 1) * tau, number_batches])

    for l in range(number_batches):
        H0pNt = Y[0:(p + 1) * output_dimension, :, 0, l]

        for k in range(max_k + 2 + (xi - 1) * tau):
            # Build Hk
            HkpNt = Y[0:(p + 1) * output_dimension, :, k, l]
            Hpnt[:, :, k, l] = HkpNt

            # Build Rk
            Rkt[:, :, k, l] = np.matmul(HkpNt, H0pNt.T)



    Hkxzt = np.zeros([xi * (p + 1) * output_dimension, number_batches * (p + 1) * output_dimension, max_k + 2])



    # Build Hk1
    Hk1t = np.zeros([xi * (p + 1) * output_dimension, number_batches * (p + 1) * output_dimension])
    for i in range(xi):
        for j in range(number_batches):
            Hk1t[i * (p + 1) * output_dimension:(i + 1) * (p + 1) * output_dimension, j * (p + 1) * output_dimension:(j + 1) * (p + 1) * output_dimension] = Rkt[:, :, 0 + i * tau, j]

    Hkxzt[:, :, 0] = Hk1t


    for k in range(max_k + 1):

        # Build Hk2
        Hk2t = np.zeros([xi * (p + 1) * output_dimension, number_batches * (p + 1) * output_dimension])
        for i in range(xi):
            for j in range(number_batches):
                Hk2t[i * (p + 1) * output_dimension:(i + 1) * (p + 1) * output_dimension, j * (p + 1) * output_dimension:(j + 1) * (p + 1) * output_dimension] = Rkt[:, :, k + i * tau + 1, j]

        Hkxzt[:, :, k + 1] = Hk2t

        # SVD H
        (R1, sigma1, St1) = LA.svd(Hk1t, full_matrices=True)
        Sigma1 = np.diag(sigma1)
        sigma.append(sigma1)

        # SVD Y2
        (R2, sigma2, St2) = LA.svd(Hk2t, full_matrices=True)
        Sigma2 = np.diag(sigma2)

        # Calculating Hk1 for the next time step
        Hk1t = Hk2t

        # Applying state_dim
        Rn1 = R1[:, 0:state_dimension]
        Snt1 = St1[0:state_dimension, :]
        Sigman1 = Sigma1[0:state_dimension, 0:state_dimension]
        Rn2 = R2[:, 0:state_dimension]
        Snt2 = St2[0:state_dimension, :]
        Sigman2 = Sigma2[0:state_dimension, 0:state_dimension]

        # Observability matrix at k and k+1 and state ensemble variable matrix at k and k+1
        O1 = np.matmul(Rn1, LA.sqrtm(Sigman1))
        X1 = np.matmul(LA.sqrtm(Sigman1), Snt1)
        O2 = np.matmul(Rn2, LA.sqrtm(Sigman2))
        X2 = np.matmul(LA.sqrtm(Sigman2), Snt2)

        # Store observability matrices
        Ok[:, :, k] = O1[0:xi * p * output_dimension, :]
        Ok1[:, :, k] = O2[0:xi * p * output_dimension, :]


        # Identified matrices
        A_id[:, :, k] = np.matmul(X2, LA.pinv(X1))
        C_id[:, :, k] = O1[0:output_dimension, :]


    # Create corresponding functions
    def A(tk):
        return A_id[:, :, int(round(tk * frequency))]

    def B(tk):
        return B_id[:, :, int(round(tk * frequency))]

    def C(tk):
        return C_id[:, :, int(round(tk * frequency))]

    def D(tk):
        return D_id[:, :, int(round(tk * frequency))]


    # Get x0
    x0 = identificationInitialCondition(full_experiment.input_signals[0], full_experiment.output_signals[0], A, B, C, D, 0, p)


    return A, B, C, D, x0, Ok, Ok1, sigma, Hpnt, Hkxzt, Rkt
