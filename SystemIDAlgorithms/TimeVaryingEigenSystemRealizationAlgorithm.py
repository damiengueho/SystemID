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


def timeVaryingEigenSystemRealizationAlgorithm(free_decay_experiments, hki, D, state_dimension, p, q, **kwargs):

    # Dimensions and number of steps
    output_dimension, input_dimension, number_steps = D.shape

    # Compute free response experiments Y matrices
    number_free_decay_experiments = free_decay_experiments.number_experiments
    free_decay_outputs = free_decay_experiments.output_signals
    Y = np.zeros([(p + 1) * output_dimension, number_free_decay_experiments, q])
    for k in range(q):
        for i in range(p + 1):
            for j in range(number_free_decay_experiments):
                Y[i * output_dimension:(i + 1) * output_dimension, j, k] = free_decay_outputs[j].data[:, i + k]


    # Frequency
    frequency = free_decay_experiments.input_signals[0].frequency


    # Applying Transformation to the first q time steps
    apply_transformation = kwargs.get('apply_transformation', False)


    # Initializing Identified matrices
    A_id = np.zeros([state_dimension, state_dimension, number_steps])
    B_id = np.zeros([state_dimension, input_dimension, number_steps])
    C_id = np.zeros([output_dimension, state_dimension, number_steps])
    D_id = np.zeros([output_dimension, input_dimension, number_steps])


    # Store observability matrices at each step (for eigenvalue check)
    Ok = np.zeros([p * output_dimension, state_dimension, number_steps])
    Ok1 = np.zeros([p * output_dimension, state_dimension, number_steps])


    # Store SVD at each time step
    sigma = []


    # Matrices D
    D_id = D


    # First Hpq1
    Hpq1 = np.zeros([(p + 1) * output_dimension, q * input_dimension])
    for i in range(p + 1):
        for j in range(q):
            Hpq1[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension] = hki[(q + i - 1) * output_dimension:(q + i) * output_dimension, (q - 1 - j) * input_dimension:(q - j) * input_dimension]

    print(Hpq1)


    # Calculating next A, B, C
    for k in range(q, number_steps - p - 1):
        Hpq2 = np.zeros([(p + 1) * output_dimension, q * input_dimension])
        for i in range(p + 1):
            for j in range(q):
                Hpq2[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension] = hki[(k + 1 + i - 1) * output_dimension:(k + 1 + i) * output_dimension, (k + 1 - 1 - j) * input_dimension:(k + 1 - j) * input_dimension]

        # SVD Hpq1
        (R1, sigma1, St1) = LA.svd(Hpq1, full_matrices=True)
        Sigma1 = np.diag(sigma1)
        sigma.append(sigma1)

        # SVD Hpq2
        (R2, sigma2, St2) = LA.svd(Hpq2, full_matrices=True)
        Sigma2 = np.diag(sigma2)

        # Calculating Hpq1 for next time step
        Hpq1 = Hpq2

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

        # Identified matrices
        A_id[:, :, k] = np.matmul(LA.pinv(O22), O1[output_dimension:, :])
        B_id[:, :, k] = R1[:, 0:input_dimension]
        C_id[:, :, k] = O1[0:output_dimension, :]


    # Calculating first q A, B, C - Free Response
    for k in range(q):
        # SVD Y1
        (R1, sigma1, St1) = LA.svd(Y[0:p * output_dimension, :, k], full_matrices=True)
        Sigma1 = np.diag(sigma1)
        sigma.append(sigma1)

        # SVD Y2
        (R2, sigma2, St2) = LA.svd(Y[output_dimension:(p + 1) * output_dimension, :, k], full_matrices=True)
        Sigma2 = np.diag(sigma2)

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
        Ok[:, :, k] = O1
        Ok1[:, :, k] = O2

        # Calculating corresponding Hp1
        Hp1 = hki[k * output_dimension:(k + p) * output_dimension, k*input_dimension:(k + 1) * input_dimension]


        # Identified matrices
        if apply_transformation:
            Tkp1 = np.matmul(LA.pinv(Ok[:, :, q]), Ok1[:, :, k])
            Tk = np.matmul(LA.pinv(Ok[:, :, q]), Ok[:, :, k])
            A_id[:, :, k] = np.matmul(Tkp1, np.matmul(np.matmul(X2, LA.pinv(X1)), LA.pinv(Tk)))
            B_id[:, :, k] = np.matmul(Tkp1, np.matmul(LA.pinv(O2), Hp1))
            C_id[:, :, k] = np.matmul(O1[0:output_dimension, :], LA.pinv(Tk))
        else:
            A_id[:, :, k] = np.matmul(X2, LA.pinv(X1))
            B_id[:, :, k] = np.matmul(LA.pinv(O2), Hp1)
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


    # # Get x0
    # x0 = identificationInitialCondition(full_experiment.input_signals[0], full_experiment.output_signals[0], A, B, C, D, 0, p)
    #
    #
    # # Get xq
    # full_experiment_input_q = DiscreteSignal(full_experiment.input_signals[0].dimension,
    #                                          full_experiment.input_signals[0].total_time - q * full_experiment.input_signals[0].dt,
    #                                          full_experiment.input_signals[0].frequency, signal_shape='External', data=full_experiment.input_signals[0].data[:, q:])
    # full_experiment_output_q = DiscreteSignal(full_experiment.output_signals[0].dimension,
    #                                          full_experiment.output_signals[0].total_time - q * full_experiment.output_signals[0].dt,
    #                                          full_experiment.output_signals[0].frequency, signal_shape='External', data=full_experiment.output_signals[0].data[:, q:])
    # xq = identificationInitialCondition(full_experiment_input_q, full_experiment_output_q, A, B, C, D, q * full_experiment.input_signals[0].dt, q)


    return A, B, C, D, Ok, Ok1, sigma
