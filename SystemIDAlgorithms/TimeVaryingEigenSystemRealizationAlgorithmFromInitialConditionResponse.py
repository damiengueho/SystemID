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

from SystemIDAlgorithms.IdentificationInitialCondition import identificationInitialCondition



def timeVaryingEigenSystemRealizationAlgorithmFromInitialConditionResponse(free_decay_experiments, state_dimension, p):

    # Dimensions and number of steps
    input_dimension = free_decay_experiments.input_dimension
    output_dimension = free_decay_experiments.output_dimension
    number_free_decay_experiments = free_decay_experiments.number_experiments
    number_steps = free_decay_experiments.output_signals[0].number_steps


    # Frequency
    frequency = free_decay_experiments.input_signals[0].frequency


    # Initializing Identified matrices
    A_id = np.zeros([state_dimension, state_dimension, number_steps])
    B_id = np.zeros([state_dimension, input_dimension, number_steps])
    C_id = np.zeros([output_dimension, state_dimension, number_steps])
    D_id = np.zeros([output_dimension, input_dimension, number_steps])


    # Store observability matrices at each step (for eigenvalue check)
    Ok = np.zeros([p * output_dimension, state_dimension, number_steps])
    Ok1 = np.zeros([p * output_dimension, state_dimension, number_steps])


    # Store Singular Values
    Sigma = []


    # Construct Y
    Y = np.zeros([(p + 1) * output_dimension, number_free_decay_experiments, number_steps - p])
    for k in range(number_steps - p):
        for i in range(p + 1):
            for j in range(number_free_decay_experiments):
                Y[i * output_dimension:(i + 1) * output_dimension, j, k] = free_decay_experiments.output_signals[j].data[:, i + k]


    # Calculating A and C matrices - Free Response
    for k in range(number_steps - p):
        # SVD Y1
        (R1, sigma1, St1) = LA.svd(Y[0:p * output_dimension, :, k], full_matrices=True)
        Sigma1 = np.diag(sigma1)
        Sigma.append(sigma1)

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

        # ICs
        if k == 0:
            X0 = X1

        # Store observability matrices
        Ok[:, :, k] = O1
        Ok1[:, :, k] = O2

        # A and C matrices
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
    # x0 = identificationInitialCondition(full_experiment.input_signals[0], full_experiment.output_signals[0], A, B, C, D, 0, p)


    return A, B, C, D, Ok, Ok1, Sigma, X0
