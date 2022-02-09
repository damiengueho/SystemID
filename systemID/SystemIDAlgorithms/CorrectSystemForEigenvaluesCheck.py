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

from systemID.SystemIDAlgorithms.GetObservabilityMatrix import getObservabilityMatrix
from systemID.ClassesGeneral.ClassSystem import DiscreteLinearSystem


def correctSystemForEigenvaluesCheck(system, number_steps, p):
    """
    Purpose:
        Correct the system matrices :math:`A_k` of a system with a matrix multiplication on the left by \
        :math:`{\\boldsymbol{O}_k^{(p)}}^\dagger \\boldsymbol{O}_{k+1}^{(p)}`, where :math:`\\boldsymbol{O}_k^{(p)}` \
        is the observability matrix at time :math:`k` of the coresponding linear time-varying system.

    Parameters:
        - **system** (``DiscreteLinearSystem``): the system to be corrected.
        - **number_steps** (``int``): the number of steps for which the correction will be made.
        - **p** (``int``): the block size of the observability matrix .

    Returns:
        - **corrected_system** (``DiscreteLinearSystem``): the corrected system.

    Imports:
        - ``import numpy as np``
        - ``import scipy.linalg as LA``
        - ``from systemID.SystemIDAlgorithms.GetObservabilityMatrix import getObservabilityMatrix``
        - ``from systemID.ClassesGeneral.ClassSystem import DiscreteLinearSystem``

    Description:
        This multiplicative correction on the left represents one part of the similarity transform that exists between two topologically equivalent \
        realizations. If :math:`A_k` and :math:`\\hat{A}_k` represent the system matrices of two topologically equivalent \
        realizations, then the matrices

        .. math::
            :nowrap:

                \\begin{align}
                    {} & {\\boldsymbol{O}_k^{(p)}}^\dagger \\boldsymbol{O}_{k+1}^{(p)}A_k, \\\\
                    {} & {\\hat{\\boldsymbol{O}}_k^{(p)}}^\dagger \\hat{\\boldsymbol{O}}_{k+1}^{(p)}\\hat{A}_k,
                \\end{align}

        have the same eigenvalues. The program calculates the observability matrices :math:`{\\boldsymbol{O}_k^{(p)}}` \
        and :math:`\\boldsymbol{O}_{k+1}^{(p)}` at each time step from :math:`k = 0` to **number_steps** \
        and multiply :math:`A_k` on the left by :math:`{\\boldsymbol{O}_k^{(p)}}^\dagger \\boldsymbol{O}_{k+1}^{(p)}`.

    See Also:
        - :py:mod:`~SystemIDAlgorithms.GetObservabilityMatrix.getObservabilityMatrix`
        - :py:mod:`~ClassesGeneral.ClassSystem.DiscreteLinearSystem`
    """

    # Dimension and parameters
    state_dimension, _ = system.A(0).shape
    dt = system.dt
    frequency = system.frequency

    # Initialize corrected A
    A_corrected_matrix = np.zeros([state_dimension, state_dimension, number_steps])

    # Apply correction
    for i in range(number_steps):
        O1 = getObservabilityMatrix(system.A, system.C, p, i * dt, dt)
        O2 = getObservabilityMatrix(system.A, system.C, p, (i + 1) * dt, dt)
        A_corrected_matrix[:, :, i] = np.matmul(LA.pinv(O1), np.matmul(O2, system.A(i * dt)))

    def A_corrected(tk):
        return A_corrected_matrix[:, :, int(round(tk * frequency))]

    return DiscreteLinearSystem(system.frequency, system.state_dimension, system.input_dimension, system.output_dimension, system.initial_states, system.name + 'corrected', A_corrected, system.B, system.C, system.D)
