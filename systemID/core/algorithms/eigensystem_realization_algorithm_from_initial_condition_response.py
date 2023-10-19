"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""


import numpy
import scipy


def eigensystem_realization_algorithm_from_initial_condition_response(output_data: numpy.ndarray, state_dimension: int, p: int = None, q: int = None):
    """
    Purpose:
        Compute a balanced state-space realization :math:`(\hat{A}, \hat{C})` of a linear time-invariant
        system from a set of output data :math:`\{\\boldsymbol{y}^{\# i}\}_{i=1..N}`.


    Parameters:
        - **output_signals** (``list``): a list of ``DiscreteSignals``.
        - **state_dimension** (``int``): the dimension, :math:`n`, of the balanced realization (most observable subspace).
        - **p** (``int``, optional): the number of row blocks of the Hankel matrices. If not specified, :math:`p=\\lfloor (N-1)/2\\rfloor`.
        - **q** (``int``, optional): the number of column blocks of the Hankel matrices. If not specified, :math:`q=\min(p, \\lfloor (N-1)/2\\rfloor)`.

    Returns:
        - **A** (``fun``): the identified system matrix :math:`\hat{A}`.
        - **C** (``fun``): the identified output influence matrix :math:`\hat{C}`.
        - **X0** (``np.array``): the set of identified initial conditions corresponding to each signals from **output_signals**. Each column corresponds to one initial condition.
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
        - :py:mod:`~SystemIDAlgorithms.EigenSystemRealizationAlgorithmWithDataCorrelation.eigenSystemRealizationAlgorithmWithDataCorrelation`
        - :py:mod:`~SystemIDAlgorithms.EigenSystemRealizationAlgorithmWithDataCorrelationFromInitialConditionResponse.eigenSystemRealizationAlgorithmWithDataCorrelationFromInitialConditionResponse`
    """

    # Return
    results = {}

    # Number of signals
    (output_dimension, number_steps, number_experiments) = output_data.shape

    # Building pseudo Markov parameters
    markov_parameters = []
    for i in range(number_steps):
        Yk = numpy.zeros([output_dimension, number_experiments])
        for j in range(number_experiments):
            Yk[:, j] = output_data[:, i, j]
        markov_parameters.append(Yk)

    # Sizes
    max_size = int(numpy.floor((len(markov_parameters) - 1) / 2))
    if p is None:
        p = max_size
    else:
        p = min(p, max_size)
    if q is None:
        q = max_size
    else:
        q = min(q, max_size)

    # Hankel matrices H(0) and H(1)
    H0 = numpy.zeros([p * output_dimension, q * number_experiments])
    H1 = numpy.zeros([p * output_dimension, q * number_experiments])
    for i in range(p):
        for j in range(q):
            H0[i * output_dimension:(i + 1) * output_dimension, j * number_experiments:(j + 1) * number_experiments] = markov_parameters[i + j]
            H1[i * output_dimension:(i + 1) * output_dimension, j * number_experiments:(j + 1) * number_experiments] = markov_parameters[i + j + 1]

    # SVD H(0)
    (R, sigma, St) = scipy.linalg.svd(H0, full_matrices=True)
    Sigma = numpy.diag(sigma)

    # # MAC and MSV
    # mac_msv = kwargs.get('mac_msv', False)
    # if mac_msv:
    #     pm, qr = H0.shape
    #     n = min(pm, qr)
    #     Rn = R[:, 0:n]
    #     Snt = St[0:n, :]
    #     Sigman = Sigma[0:n, 0:n]
    #     Op = numpy.matmul(Rn, LA.sqrtm(Sigman))
    #     Rq = numpy.matmul(LA.sqrtm(Sigman), Snt)
    #     A_id = numpy.matmul(LA.pinv(Op), numpy.matmul(H1, LA.pinv(Rq)))
    #     B_id = Rq[:, 0:input_dimension]
    #     C_id = Op[0:output_dimension, :]
    #     MAC, MSV = mac_and_msv(A_id, B_id, C_id, Rq, p)
    # else:
    #     MAC = []
    #     MSV = []

    # Matrices Rn, Sn, Sigman
    Rn = R[:, 0:state_dimension]
    Snt = St[0:state_dimension, :]
    Sigman = Sigma[0:state_dimension, 0:state_dimension]

    # Identified matrices
    Op = numpy.matmul(Rn, scipy.linalg.sqrtm(Sigman))
    Rq = numpy.matmul(scipy.linalg.sqrtm(Sigman), Snt)
    A_id = numpy.matmul(scipy.linalg.pinv(Op), numpy.matmul(H1, scipy.linalg.pinv(Rq)))
    X0 = Rq[:, 0:number_experiments]
    C_id = Op[0:output_dimension, :]


    def A(tk):
        return A_id

    def C(tk):
        return C_id

    results['A'] = A
    results['C'] = C
    results['X0'] = X0
    results['H0'] = H0
    results['H1'] = H1
    results['R'] = R
    results['Sigma'] = Sigma
    results['St'] = St
    results['Rn'] = Rn
    results['Sigman'] = Sigman
    results['Snt'] = Snt
    results['Op'] = Op
    results['Rq'] = Rq


    return results
