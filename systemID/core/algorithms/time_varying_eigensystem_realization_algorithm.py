"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 25
"""


import numpy
import scipy

from systemID.core.algorithms.time_varying_eigensystem_realization_algorithm_from_initial_condition_response import time_varying_eigensystem_realization_algorithm_from_initial_condition_response

def time_varying_eigensystem_realization_algorithm(hki: numpy.ndarray,
                                                   D: numpy.ndarray,
                                                   state_dimension: int,
                                                   dt: float,
                                                   free_response_data: numpy.ndarray = None,
                                                   p: int = None,
                                                   q: int = None,
                                                   apply_transformation: bool = False,
                                                   show_progress: bool = False):
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

    # Return
    results = {}

    # # MAC and MSV
    # mac_msv = kwargs.get('mac_msv', False)

    # Dimensions and number of steps
    output_dimension, input_dimension, number_steps = D.shape

    # Initializing Identified matrices
    A_id = numpy.zeros([state_dimension, state_dimension, number_steps])
    B_id = numpy.zeros([state_dimension, input_dimension, number_steps])
    C_id = numpy.zeros([output_dimension, state_dimension, number_steps])
    D_id = D

    # Store observability matrices at each step (for eigenvalue check)
    Ok = numpy.zeros([p * output_dimension, state_dimension, number_steps])
    Ok1 = numpy.zeros([p * output_dimension, state_dimension, number_steps])

    # Store Singular Values
    Sigma = []

    # Compute first few time steps using free response
    number_experiments = free_response_data.shape[2]


    # First Hpq1
    Hpq1 = numpy.zeros([(p + 1) * output_dimension, q * input_dimension])
    for i in range(p + 1):
        for j in range(q):
            Hpq1[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension] = hki[(q + i - 1) * output_dimension:(q + i) * output_dimension, (q - 1 - j) * input_dimension:(q - j) * input_dimension]

    # First step SVD
    (R1, sigma1, St1) = scipy.linalg.svd(Hpq1, full_matrices=True)

    # Calculating next A, B, C
    for k in range(q, number_steps - p - 1):

        if show_progress:
            print('Step', k + 1, 'out of', number_steps - p - 1)


        Hpq2 = numpy.zeros([(p + 1) * output_dimension, q * input_dimension])
        for i in range(p + 1):
            for j in range(q):
                Hpq2[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension] = hki[(k + 1 + i - 1) * output_dimension:(k + 1 + i) * output_dimension, (k + 1 - 1 - j) * input_dimension:(k + 1 - j) * input_dimension]

        # SVD Hpq1
        Sigma1 = numpy.diag(sigma1)
        Sigma.append(sigma1)

        # SVD Hpq2
        (R2, sigma2, St2) = scipy.linalg.svd(Hpq2, full_matrices=True)
        Sigma2 = numpy.diag(sigma2)

        # Applying state_dim
        Rn1 = R1[:, 0:state_dimension]
        Snt1 = St1[0:state_dimension, :]
        Sigman1 = Sigma1[0:state_dimension, 0:state_dimension]
        Rn2 = R2[:, 0:state_dimension]
        Snt2 = St2[0:state_dimension, :]
        Sigman2 = Sigma2[0:state_dimension, 0:state_dimension]

        # Observability matrix at k and k+1 and controllability matrix at k
        O1 = numpy.matmul(Rn1, scipy.linalg.sqrtm(Sigman1))
        O2 = numpy.matmul(Rn2, scipy.linalg.sqrtm(Sigman2))
        R1 = numpy.matmul(scipy.linalg.sqrtm(Sigman2), Snt2)

        # Deleting last block-line of observability matrices to conform with sizes
        O11 = O1[0:p * output_dimension, :]
        O22 = O2[0:p * output_dimension, :]

        # Store observability matrices
        Ok[:, :, k] = O11
        Ok1[:, :, k] = O22

        # Identified matrices
        A_id[:, :, k] = numpy.matmul(scipy.linalg.pinv(O22), O1[output_dimension:, :])
        B_id[:, :, k] = R1[:, 0:input_dimension]
        C_id[:, :, k] = O1[0:output_dimension, :]

        # Time shift
        Hpq1 = Hpq2
        (R1, sigma1, St1) = (R2, sigma2, St2)


    if free_response_data is not None:
        # Calculating first q A, B, C - Free Response

        results_tvera_ic = time_varying_eigensystem_realization_algorithm_from_initial_condition_response(output_data=free_response_data,
                                                                                                          state_dimension=state_dimension,
                                                                                                          dt=dt,
                                                                                                          p=p,
                                                                                                          max_time_step=q)

        A_id[:, :, 0:q] = results_tvera_ic['A_id'][:, :, 0:q]
        C_id[:, :, 0:q] = results_tvera_ic['C_id'][:, :, 0:q]
        Ok[:, :, 0:q] = results_tvera_ic['Ok'][:, :, 0:q]
        Ok1[:, :, 0:q] = results_tvera_ic['Ok1'][:, :, 0:q]

        Sigma = Sigma + results_tvera_ic['Sigma']

        for k in range(q):
            # Calculating corresponding Hp1
            Hp1 = hki[k * output_dimension:(k + p) * output_dimension, k*input_dimension:(k + 1) * input_dimension]

            # Identified matrices
            if apply_transformation:
                Tkp1 = numpy.matmul(scipy.linalg.pinv(Ok[:, :, q]), Ok1[:, :, k])
                Tk = numpy.matmul(scipy.linalg.pinv(Ok[:, :, q]), Ok[:, :, k])

                A_id[:, :, k] = numpy.matmul(Tkp1, numpy.matmul(A_id[:, :, k], scipy.linalg.pinv(Tk)))
                B_id[:, :, k] = numpy.matmul(Tkp1, numpy.matmul(scipy.linalg.pinv(Ok1[:, :, k]), Hp1))
                C_id[:, :, k] = numpy.matmul(C_id[:, :, k], scipy.linalg.pinv(Tk))

            else:
                B_id[:, :, k] = numpy.matmul(scipy.linalg.pinv(Ok1[:, :, k]), Hp1)

        results['X0'] = results_tvera_ic['X0']

    # Create corresponding functions
    def A(tk):
        return A_id[:, :, int(round(tk / dt))]

    def B(tk):
        return B_id[:, :, int(round(tk / dt))]

    def C(tk):
        return C_id[:, :, int(round(tk / dt))]

    def D(tk):
        return D_id[:, :, int(round(tk / dt))]

    results['A'] = A
    results['B'] = B
    results['C'] = C
    results['D'] = D
    results['Ok'] = Ok
    results['Ok1'] = Ok1
    results['Sigma'] = Sigma


    return results
