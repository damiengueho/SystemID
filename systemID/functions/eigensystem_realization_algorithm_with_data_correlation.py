"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""


import numpy as np
import scipy.linalg as LA

from systemID.functions.mac_and_msv import mac_and_msv




def eigensystem_realization_algorithm_with_data_correlation(markov_parameters, state_dimension, **kwargs):
    """
    Purpose:
        Compute a balanced state-space realization :math:`(\hat{A}, \hat{B}, \hat{C}, \hat{D})` of a linear time-invariant
        system from a set of Markov parameters :math:`\{h_i\}_{i=0..N}`. This modified version of ERA takes advantage of
        data correlation to minimize the effect of noise in the data.


    Parameters:
        - **markov_parameters** (``list``): a list of Markov parameters :math:`\{h_i\}_{i=0..N}`.
        - **state_dimension** (``int``): the dimension, :math:`n`, of the balanced realization (most observable and controllable subspace).
        - **p** (``int``, optional): the number of row blocks of the Hankel matrices. If not specified, :math:`p=\\lfloor N/2\\rfloor`.
        - **q** (``int``, optional): the number of column blocks of the Hankel matrices. If not specified, :math:`q=\\min(p, \lfloor N/2\\rfloor)`.
        - **xi** (``int``, optional):
        - **zeta** (``int``, optional):
        - **tau** (``int``, optional):

    Returns:
        - **A** (``fun``): the identified system matrix :math:`\hat{A}`.
        - **B** (``fun``): the identified input influence matrix :math:`\hat{B}`.
        - **C** (``fun``): the identified output influence matrix :math:`\hat{C}`.
        - **D** (``fun``): the identified direct transmission matrix :math:`\hat{D}`.
        - **H0** (``np.array``): the Hankel matrix :math:`H_0`.
        - **H1** (``np.array``): the Hankel matrix :math:`H_1`.
        - **R** (``np.array``): the left eigenvectors of :math:`H_0` computed through a singular value decomposition.
        - **Sigma** (``np.array``): diagonal matrix of singular values of :math:`H_0` computed through a singular value decomposition.
        - **St** (``np.array``): the right eigenvectors of :math:`H_0` computed through a singular value decomposition.
        - **Rn** (``np.array``): the first :math:`n` columns of :math:`R`.
        - **Sigman** (``np.array``): the first :math:`n` rows and :math:`n` columns of :math:`\Sigma`.
        - **Snt** (``np.array``): the first :math:`n` rows of :math:`S^T`.
        - **Op** (``np.array``): the observability matrix.
        - **Rq** (``np.array``): the controllability matrix.
        - **MAC** (``list``): MAC values.
        - **MSV** (``list``): MSV values.

    Imports:
        - ``import numpy as np``
        - ``import scipy.linalg as LA``
        - ``from systemID.SystemIDAlgorithms.GetMACandMSV import getMACandMSV``

    Description:


    See Also:
        - :py:mod:`~SystemIDAlgorithms.GetMACandMSV.getMACandMSV`
        - :py:mod:`~SystemIDAlgorithms.EigenSystemRealizationAlgorithm.eigenSystemRealizationAlgorithm`
        - :py:mod:`~SystemIDAlgorithms.EigenSystemRealizationAlgorithmFromInitialConditionResponse.eigenSystemRealizationAlgorithmFromInitialConditionResponse`
        - :py:mod:`~SystemIDAlgorithms.EigenSystemRealizationAlgorithmWithDataCorrelationFromInitialConditionResponse.eigenSystemRealizationAlgorithmWithDataCorrelationFromInitialConditionResponse`
    """

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

    # Dimensions
    (output_dimension, input_dimension) = markov_parameters[0].shape

    # Hankel matrices
    H = np.zeros([p * output_dimension, q * input_dimension, gamma + 1])
    for i in range(p):
        for j in range(q):
            for k in range(gamma + 1):
                H[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension, k] = markov_parameters[i + j + 1 + k]

    # Data Correlation Matrices
    HR = np.zeros([p * output_dimension, p * output_dimension, gamma + 1])
    for i in range(gamma + 1):
        HR[:, :, i] = np.matmul(H[:, :, i], np.transpose(H[:, :, 0]))

    # Building Block Correlation Hankel Matrices
    H0 = np.zeros([(xi + 1) * p * output_dimension, (zeta + 1) * p * output_dimension])
    H1 = np.zeros([(xi + 1) * p * output_dimension, (zeta + 1) * p * output_dimension])
    for i in range(xi + 1):
        for j in range(zeta + 1):
            H0[i * p * output_dimension:(i + 1) * p * output_dimension, j * p * output_dimension:(j + 1) * p * output_dimension] = HR[:, :, (i + j) * tau]
            H1[i * p * output_dimension:(i + 1) * p * output_dimension, j * p * output_dimension:(j + 1) * p * output_dimension] = HR[:, :, (i + j) * tau + 1]

    # SVD H(0)
    (R, sigma, St) = LA.svd(H0, full_matrices=True)
    Sigma = np.diag(sigma)

    # MAC and MSV
    mac_msv = kwargs.get('mac_msv', False)
    if mac_msv:
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
        MAC, MSV = mac_and_msv(A_id, B_id, C_id, Rq, p)
    else:
        MAC = []
        MSV = []

    # Matrices Rn, Sn, Sigman
    Rn = R[:, 0:state_dimension]
    Snt = St[0:state_dimension, :]
    Sigman = Sigma[0:state_dimension, 0:state_dimension]

    # Identified matrices
    Op = np.matmul(Rn, LA.sqrtm(Sigman))
    Rq = np.matmul(LA.sqrtm(Sigman), Snt)
    A_id = np.matmul(LA.pinv(Op), np.matmul(H1, LA.pinv(Rq)))
    B_id = np.matmul(LA.pinv(Op[0:p * output_dimension, :]), H[:, :, 0])[:, 0:input_dimension]
    C_id = Op[0:output_dimension, :]
    D_id = markov_parameters[0]

    def A(tk):
        return A_id

    def B(tk):
        return B_id

    def C(tk):
        return C_id

    def D(tk):
        return D_id

    return A, B, C, D, H0, H1, R, Sigma, St, Rn, Sigman, Snt, Op, Rq, MAC, MSV
