"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""



import numpy
import scipy.linalg as LA
from systemID.functions.mac_and_msv import mac_and_msv


def time_varying_eigensystem_realization_algorithm_from_initial_condition_response(output_data: numpy.ndarray,
                                                                                   state_dimension: int,
                                                                                   dt: float,
                                                                                   p: int,
                                                                                   max_time_step: int = None
):
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

    # Number of Signals
    (output_dimension, number_steps, number_experiments) = output_data.shape

    # Max time step
    if max_time_step == None:
        max_time_step = number_steps - p
    else:
        max_time_step = min(max_time_step, number_steps - p)

    # Initializing Identified matrices
    A_id = numpy.zeros([state_dimension, state_dimension, max_time_step])
    C_id = numpy.zeros([output_dimension, state_dimension, max_time_step])

    # Store observability matrices at each step (for eigenvalue check)
    Ok = numpy.zeros([p * output_dimension, state_dimension, max_time_step])
    Ok1 = numpy.zeros([p * output_dimension, state_dimension, max_time_step])

    # Store Singular Values
    Sigma = []

    # # MAC and MSV
    # mac_msv = kwargs.get('mac_msv', False)
    # MAC = []
    # MSV = []

    # Construct Y
    Y = numpy.zeros([(max_time_step + p) * output_dimension, number_experiments])
    for j in range(number_experiments):
        Y[:, j] = output_data[:, 0:max_time_step + p, j].reshape((max_time_step + p) * output_dimension, order='F')

    # Calculating for first time step
    (R1, sigma1, St1) = LA.svd(Y[0:p * output_dimension, :], full_matrices=True)

    # Calculating A and C matrices
    for k in range(max_time_step):

        # SVD Y1
        Sigma1 = numpy.diag(sigma1)
        Sigma.append(sigma1)

        # SVD Y2
        (R2, sigma2, St2) = LA.svd(Y[output_dimension + k * output_dimension:(p + 1) * output_dimension + k * output_dimension, :], full_matrices=True)
        Sigma2 = numpy.diag(sigma2)

        # if mac_msv:
        #     pm, qr = Y[0:p * output_dimension, :].shape
        #     n = min(pm, qr)
        #     Rn = R1[:, 0:n]
        #     Snt = St1[0:n, :]
        #     Sigman = Sigma1[0:n, 0:n]
        #     Op = np.matmul(Rn, LA.sqrtm(Sigman))
        #     Rq = np.matmul(LA.sqrtm(Sigman), Snt)
        #     A_idt = np.matmul(LA.pinv(Op), np.matmul(Y[output_dimension + k * output_dimension:(p + 1) * output_dimension + k * output_dimension, :], LA.pinv(Rq)))
        #     B_idt = Rq[:, 0:input_dimension]
        #     C_idt = Op[0:output_dimension, :]
        #     mac, msv = mac_and_msv(A_idt, B_idt, C_idt, Rq, p)
        #     MAC.append(mac)
        #     MSV.append(msv)

        # Applying state_dim
        Rn1 = R1[:, 0:state_dimension]
        Snt1 = St1[0:state_dimension, :]
        Sigman1 = Sigma1[0:state_dimension, 0:state_dimension]
        Rn2 = R2[:, 0:state_dimension]
        Snt2 = St2[0:state_dimension, :]
        Sigman2 = Sigma2[0:state_dimension, 0:state_dimension]

        # Observability matrix at k and k+1 and state ensemble variable matrix at k and k+1
        O1 = numpy.matmul(Rn1, LA.sqrtm(Sigman1))
        X1 = numpy.matmul(LA.sqrtm(Sigman1), Snt1)
        O2 = numpy.matmul(Rn2, LA.sqrtm(Sigman2))
        X2 = numpy.matmul(LA.sqrtm(Sigman2), Snt2)

        # ICs
        if k == 0:
            X0 = X1

        # Store observability matrices
        Ok[:, :, k] = O1
        Ok1[:, :, k] = O2

        # A and C matrices
        A_id[:, :, k] = numpy.matmul(X2, LA.pinv(X1))
        C_id[:, :, k] = O1[0:output_dimension, :]

        # Time shift
        (R1, sigma1, St1) = (R2, sigma2, St2)


    # Create corresponding functions
    def A(tk):
        return A_id[:, :, int(round(tk / dt))]

    def C(tk):
        return C_id[:, :, int(round(tk / dt))]

    results['A'] = A
    results['A_id'] = A_id
    results['C'] = C
    results['C_id'] = C_id
    results['X0'] = X0
    results['Ok'] = Ok
    results['Ok1'] = Ok1
    results['Sigma'] = Sigma
    results['Y'] = Y

    return results
