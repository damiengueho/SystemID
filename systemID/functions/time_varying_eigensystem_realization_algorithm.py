"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""



import numpy as np
import scipy.linalg as LA

from systemID.functions.time_varying_eigensystem_realization_algorithm_from_initial_condition_response import time_varying_eigensystem_realization_algorithm_from_initial_condition_response

def time_varying_eigensystem_realization_algorithm(hki, D, frequency, state_dimension, p, q, **kwargs):
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

    # Show progression
    show_progress = kwargs.get('show_progress', False)

    # MAC and MSV
    mac_msv = kwargs.get('mac_msv', False)

    # Applying Transformation to the first q time steps
    apply_transformation = kwargs.get('apply_transformation', False)

    # Dimensions and number of steps
    output_dimension, input_dimension, number_steps = D.shape

    # Initializing Identified matrices
    A_id = np.zeros([state_dimension, state_dimension, number_steps])
    B_id = np.zeros([state_dimension, input_dimension, number_steps])
    C_id = np.zeros([output_dimension, state_dimension, number_steps])
    D_id = D

    # Store observability matrices at each step (for eigenvalue check)
    Ok = np.zeros([p * output_dimension, state_dimension, number_steps])
    Ok1 = np.zeros([p * output_dimension, state_dimension, number_steps])

    # Store Singular Values
    Sigma = []

    # Compute first few time steps using free response
    free_response_signals = kwargs.get('free_response_signals', [])
    number_experiments = len(free_response_signals)


    # First Hpq1
    Hpq1 = np.zeros([(p + 1) * output_dimension, q * input_dimension])
    for i in range(p + 1):
        for j in range(q):
            Hpq1[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension] = hki[(q + i - 1) * output_dimension:(q + i) * output_dimension, (q - 1 - j) * input_dimension:(q - j) * input_dimension]

    # First step SVD
    (R1, sigma1, St1) = LA.svd(Hpq1, full_matrices=True)

    # Calculating next A, B, C
    for k in range(q, number_steps - p - 1):

        if show_progress:
            print('Step', k + 1, 'out of', number_steps - p - 1)


        Hpq2 = np.zeros([(p + 1) * output_dimension, q * input_dimension])
        for i in range(p + 1):
            for j in range(q):
                Hpq2[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension] = hki[(k + 1 + i - 1) * output_dimension:(k + 1 + i) * output_dimension, (k + 1 - 1 - j) * input_dimension:(k + 1 - j) * input_dimension]

        # SVD Hpq1
        Sigma1 = np.diag(sigma1)
        Sigma.append(sigma1)

        # SVD Hpq2
        (R2, sigma2, St2) = LA.svd(Hpq2, full_matrices=True)
        Sigma2 = np.diag(sigma2)

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

        # Time shift
        Hpq1 = Hpq2
        (R1, sigma1, St1) = (R2, sigma2, St2)


    X0_tvera_ic = 'N/A'
    # Calculating first q A, B, C - Free Response
    if number_experiments > 0:
        _, _, Ok_tvera_ic, Ok1_tvera_ic, Sigma_tvera_ic, X0_tvera_ic, A_id_tvera_ic, C_id_tvera_ic, MAC_tvera_ic, MSV_tvera_ic, Y = time_varying_eigensystem_realization_algorithm_from_initial_condition_response(free_response_signals, state_dimension, p, show_progress=show_progress, mac_msv=mac_msv, max_time_step=q)

        A_id[:, :, 0:q] = A_id_tvera_ic[:, :, 0:q]
        C_id[:, :, 0:q] = C_id_tvera_ic[:, :, 0:q]
        Ok[:, :, 0:q] = Ok_tvera_ic[:, :, 0:q]
        Ok1[:, :, 0:q] = Ok1_tvera_ic[:, :, 0:q]

        Sigma = Sigma + Sigma_tvera_ic

        for k in range(q):
            # Calculating corresponding Hp1
            Hp1 = hki[k * output_dimension:(k + p) * output_dimension, k*input_dimension:(k + 1) * input_dimension]

            # Identified matrices
            if apply_transformation:
                Tkp1 = np.matmul(LA.pinv(Ok[:, :, q]), Ok1[:, :, k])
                Tk = np.matmul(LA.pinv(Ok[:, :, q]), Ok[:, :, k])

                A_id[:, :, k] = np.matmul(Tkp1, np.matmul(A_id[:, :, k], LA.pinv(Tk)))
                B_id[:, :, k] = np.matmul(Tkp1, np.matmul(LA.pinv(Ok1[:, :, k]), Hp1))
                C_id[:, :, k] = np.matmul(C_id[:, :, k], LA.pinv(Tk))

            else:
                B_id[:, :, k] = np.matmul(LA.pinv(Ok1[:, :, k]), Hp1)


    # Create corresponding functions
    def A(tk):
        return A_id[:, :, int(round(tk * frequency))]

    def B(tk):
        return B_id[:, :, int(round(tk * frequency))]

    def C(tk):
        return C_id[:, :, int(round(tk * frequency))]

    def D(tk):
        return D_id[:, :, int(round(tk * frequency))]


    return A, B, C, D, Ok, Ok1, Sigma, X0_tvera_ic, A_id, B_id, C_id, D_id
