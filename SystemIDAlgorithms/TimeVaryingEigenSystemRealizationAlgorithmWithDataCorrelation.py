"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""



import numpy as np
import scipy.linalg as LA

from ClassesGeneral.ClassSignal import DiscreteSignal
from SystemIDAlgorithms.IdentificationInitialCondition import identificationInitialCondition


def timeVaryingEigenSystemRealizationAlgorithmWithDataCorrelation(free_decay_experiments, hki, D, full_experiment, state_dimension, **kwargs):

    # Dimensions and number of steps
    output_dimension, input_dimension, number_steps = D.shape
    number_free_decay_experiments = free_decay_experiments[0].number_experiments

    # Sizes
    p = int(kwargs.get('p', 100))
    p = min(p, 100)
    q = int(kwargs.get('q', 100))
    q = min(q, 100)
    xi = int(kwargs.get('xi', 100))
    xi = min(xi, 100)
    zeta = int(kwargs.get('zeta', 100))
    zeta = min(zeta, 100)
    tau = int(kwargs.get('tau', 100))
    tau = min(tau, 100)
    max_k = number_steps - p - (xi - 1) * tau - 2

    # Compute free response experiments Y matrices
    number_batches = len(free_decay_experiments)
    Y = np.zeros([(p + 1) * output_dimension, number_free_decay_experiments, (xi + zeta - 2) * tau + q + 1, number_batches])
    for l in range(number_batches):
        free_decay_outputs = free_decay_experiments[l].output_signals
        for k in range((xi + zeta - 2) * tau + q + 1):
            for i in range(p + 1):
                for j in range(number_free_decay_experiments):
                    Y[i * output_dimension:(i + 1) * output_dimension, j, k, l] = free_decay_outputs[j].data[:, i + k]


    # Frequency
    frequency = full_experiment.input_signals[0].frequency


    # Applying Transformation to the first q time steps
    apply_transformation = kwargs.get('apply_transformation', False)


    # Initializing Identified matrices
    A_id = np.zeros([state_dimension, state_dimension, number_steps])
    B_id = np.zeros([state_dimension, input_dimension, number_steps])
    C_id = np.zeros([output_dimension, state_dimension, number_steps])
    D_id = np.zeros([output_dimension, input_dimension, number_steps])

    # Store Generalized data correlation matrices
    Rkt = np.zeros([(p + 1) * output_dimension, (p + 1) * output_dimension, (xi + zeta - 2) * tau + q + 1, number_batches])
    Rk = np.zeros([(p + 1) * output_dimension, (p + 1) * output_dimension, max_k - (zeta - 1) * tau - q + 2, xi, zeta])


    # Store observability matrices at each step (for eigenvalue check)
    Ok = np.zeros([p * output_dimension, state_dimension, number_steps])
    Ok1 = np.zeros([p * output_dimension, state_dimension, number_steps])


    # Store SVD at each time step
    sigma = []


    # Matrices D
    D_id = D


    # Hq
    Hq = np.zeros([(p + 1) * output_dimension, q * input_dimension])
    for i in range(p + 1):
        for j in range(q):
            Hq[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension] = hki[(q + i - 1) * output_dimension:(q + i) * output_dimension, (q - 1 - j) * input_dimension:(q - j) * input_dimension]

    # Calculating Rk
    for k in range((zeta - 1) * tau + q, max_k + 2):
        for r in range(xi):
            for s in range(zeta):
                Hk = np.zeros([(p + 1) * output_dimension, q * input_dimension])
                for i in range(p + 1):
                    for j in range(q):
                        Hk[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension] = hki[(k + r + i - 1) * output_dimension:(k + r + i) * output_dimension, (k - 1 - j - s) * input_dimension:(k - j - s) * input_dimension]

                Rk[:, :, k - (zeta - 1) * tau - q, r, s] = np.matmul(Hk, Hq.T)


    # Calculating first Hxz
    Hxz1 = np.zeros([xi * (p + 1) * output_dimension, zeta * (p + 1) * output_dimension])
    for i in range(xi):
        for j in range(zeta):
            Hxz1[i * (p + 1) * output_dimension:(i + 1) * (p + 1) * output_dimension, j * (p + 1) * output_dimension:(j + 1) * (p + 1) * output_dimension] = Rk[:, :, 0, i, j]

    Hkxz = np.zeros([xi * (p + 1) * output_dimension, zeta * (p + 1) * output_dimension, max_k - (zeta - 1) * tau - q + 1])

    # Calculating next A, B, C
    for k in range((zeta - 1) * tau + q, max_k + 1):
        Hxz2 = np.zeros([xi * (p + 1) * output_dimension, zeta * (p + 1) * output_dimension])
        for i in range(xi):
            for j in range(zeta):
                Hxz2[i * (p + 1) * output_dimension:(i + 1) * (p + 1) * output_dimension, j * (p + 1) * output_dimension:(j + 1) * (p + 1) * output_dimension] = Rk[:, :, k - (zeta - 1) * tau - q + 1, i, j]

        # SVD Hpq1
        (R1, sigma1, St1) = LA.svd(Hxz1, full_matrices=True)
        Sigma1 = np.diag(sigma1)
        sigma.append(sigma1)

        # SVD Hpq2
        (R2, sigma2, St2) = LA.svd(Hxz2, full_matrices=True)
        Sigma2 = np.diag(sigma2)

        # Calculating Hpq1 for next time step
        Hkxz[:, :, k - (zeta - 1) * tau - q] = Hxz1
        Hxz1 = Hxz2

        # Applying state_dim
        Rn1 = R1[:, 0:state_dimension]
        Snt1 = St1[0:state_dimension, :]
        Sigman1 = Sigma1[0:state_dimension, 0:state_dimension]
        Rn2 = R2[:, 0:state_dimension]
        Snt2 = St2[0:state_dimension, :]
        Sigman2 = Sigma2[0:state_dimension, 0:state_dimension]

        # Observability matrix at k and k+1 and controllability matrix at k
        O1 = np.matmul(Rn1, LA.sqrtm(Sigman1))
        O2 = np.matmul(Rn2, LA.sqrtm(Sigman2))
        R1 = np.matmul(LA.sqrtm(Sigman2), Snt2)

        # Deleting last block-line of observability matrices to conform with sizes
        O11 = O1[0:p * output_dimension, :]
        O22 = O2[0:p * output_dimension, :]

        # Store observability matrices
        Ok[:, :, k] = O11
        Ok1[:, :, k] = O22

        # Calculate shifted O1
        O1aa = np.zeros([xi * p * output_dimension, state_dimension])
        O2aa = np.zeros([xi * p * output_dimension, state_dimension])
        for i in range(xi):
            O1aa[i * p * output_dimension:(i + 1) * p * output_dimension, :] = (O1[i * (p + 1) * output_dimension:(i + 1) * (p + 1) * output_dimension, :])[output_dimension:(p + 1) * output_dimension, :]
            O2aa[i * p * output_dimension:(i + 1) * p * output_dimension, :] = (O2[i * (p + 1) * output_dimension:(i + 1) * (p + 1) * output_dimension, :])[0:p * output_dimension, :]

        # Calculating Hkp1
        Hkp1 = np.zeros([p * output_dimension, q * input_dimension])
        for i in range(p):
            for j in range(q):
                Hkp1[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension] = hki[(k + 1 + i - 1) * output_dimension:(k + 1 + i) * output_dimension, (k + 1 - 1 - j) * input_dimension:(k + 1 - j) * input_dimension]


        # Identified matrices
        A_id[:, :, k] = np.matmul(LA.pinv(O2aa), O1aa)
        B_id[:, :, k] = np.matmul(LA.pinv(O22[0:p * output_dimension, :]), Hkp1)[:, 0:input_dimension]
        C_id[:, :, k] = O1[0:output_dimension, :]


    # Calculating first q A, B, C - Free Response

    # Build Generalized data correlation matrices
    Hpnt = np.zeros([(p + 1) * output_dimension, number_free_decay_experiments, q + (xi + zeta - 2) * tau + 1, number_batches])

    for l in range(number_batches):
        H0pNt = Y[0:(p + 1) * output_dimension, :, 0, l]

        for k in range((xi + zeta - 2) * tau + q + 1):
            # Build Hk
            HkpNt = Y[0:(p + 1) * output_dimension, :, k, l]
            Hpnt[:, :, k, l] = HkpNt

            # Build Rk
            Rkt[:, :, k, l] = np.matmul(HkpNt, H0pNt.T)



    Hkxzt = np.zeros([xi * (p + 1) * output_dimension, number_batches * (p + 1) * output_dimension, (zeta - 1) * tau + q + 1])

    # Build Hk1
    Hk1t = np.zeros([xi * (p + 1) * output_dimension, number_batches * (p + 1) * output_dimension])
    for i in range(xi):
        for j in range(number_batches):
            Hk1t[i * (p + 1) * output_dimension:(i + 1) * (p + 1) * output_dimension, j * (p + 1) * output_dimension:(j + 1) * (p + 1) * output_dimension] = Rkt[:, :, 0 + i * tau, j]

    Hkxzt[:, :, 0] = Hk1t

    for k in range((zeta - 1) * tau + q):

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
        Ok[:, :, k] = O1[0:p * output_dimension, :]
        Ok1[:, :, k] = O2[0:p * output_dimension, :]

        # Identified matrices
        A_id[:, :, k] = np.matmul(X2, LA.pinv(X1))
        C_id[:, :, k] = O1[0:output_dimension, :]

        # Calculating corresponding Hp1
        Hp1 = hki[k * output_dimension:(k + p + 1) * output_dimension, k * input_dimension:(k + 1) * input_dimension]


        # Identified matrices
        if apply_transformation:
            Tkp1 = np.matmul(LA.pinv(Ok[:, :, (zeta - 1) * tau + q]), Ok1[:, :, k])
            Tk = np.matmul(LA.pinv(Ok[:, :, (zeta - 1) * tau + q]), Ok[:, :, k])
            A_id[:, :, k] = np.matmul(Tkp1, np.matmul(np.matmul(X2, LA.pinv(X1)), LA.pinv(Tk)))
            B_id[:, :, k] = np.matmul(Tkp1, np.matmul(LA.pinv(O2[0:(p + 1) * output_dimension, :]), Hp1))
            C_id[:, :, k] = np.matmul(O1[0:output_dimension, :], LA.pinv(Tk))
        else:
            A_id[:, :, k] = np.matmul(X2, LA.pinv(X1))
            B_id[:, :, k] = np.matmul(LA.pinv(O2[0:(p + 1) * output_dimension, :]), Hp1)
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
    x0 = identificationInitialCondition(full_experiment.input_signals[0], full_experiment.output_signals[0], A, B, C, D, 0, q-1)


    # Get xq
    full_experiment_input_q = DiscreteSignal(full_experiment.input_signals[0].dimension,
                                             full_experiment.input_signals[0].total_time - q * full_experiment.input_signals[0].dt,
                                             full_experiment.input_signals[0].frequency, signal_shape='External', data=full_experiment.input_signals[0].data[:, q:])
    full_experiment_output_q = DiscreteSignal(full_experiment.output_signals[0].dimension,
                                             full_experiment.output_signals[0].total_time - q * full_experiment.output_signals[0].dt,
                                             full_experiment.output_signals[0].frequency, signal_shape='External', data=full_experiment.output_signals[0].data[:, q:])
    xq = identificationInitialCondition(full_experiment_input_q, full_experiment_output_q, A, B, C, D, q * full_experiment.input_signals[0].dt, q)


    return A, B, C, D, x0, xq, Ok, Ok1, sigma, Hpnt, Rkt, Hkxzt, Hkxz, Rk
