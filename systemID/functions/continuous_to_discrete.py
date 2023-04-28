"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""



import numpy as np
import scipy.linalg as LA
from scipy.integrate import quad

def continuous_to_discrete_matrices(dt, Ac, **kwargs):
    """
        Purpose:
            Provides the discrete time-invariant state matrices given the continuous time-invariant state matrices of
            a linear system.

        Parameters:
            - **dt** (``float``): the discretization time step.
            - **Ac** (``ndarray``): the continuous time-invariant state matrix.
            - **Bc** (``ndarray``, optional): the continuous time-invariant input influence matrix.
            - **expansion** (``bool``, optional):

        Returns:
            - **discrete_matrices** (``list``): a list containing the discrete time-invariant state matrices.

        Imports:
            - ``import numpy as np``
            - ``import scipy.linalg as LA``

        Description:
            Given continuous time-invariant matrices of a linear system and a step size :math:`\Delta t`, the
            corresponding discrete time-invariant matrices are calculated as

            .. math::
                :nowrap:

                    \\begin{align}
                        A &= \exp(A_c\Delta t), \\\\
                        B &= \left[A-I\right]A_c^{-1}B_c.
                    \\end{align}
    """

    discrete_matrices = []

    if (isinstance(Ac, np.ndarray) and Ac.ndim == 2 and Ac.shape[0] == Ac.shape[1]):
        A = LA.expm(dt * Ac)
        discrete_matrices.append(A)
    else:
        raise ValueError("Matrix Ac must be a 2D NumPy ndarray.")

    Bc = kwargs.get('Bc', False)
    if (isinstance(Bc, np.ndarray) and Bc.ndim == 2):
        expansion = kwargs.get('expansion', False)
        if expansion:
            def func(t):
                return LA.expm(t * Ac)
            def integrand(t, row, col):
                return func(t)[row, col]
            rows, cols = func(0).shape
            result_matrix = np.zeros((rows, cols))
            for row in range(rows):
                for col in range(cols):
                    result, _ = quad(integrand, 0, dt, args=(row, col))
                    result_matrix[row, col] = result
            discrete_matrices.append(np.matmul(result_matrix, Bc))
        else:
            discrete_matrices.append(np.matmul(A - np.eye(A.shape[0]), np.matmul(LA.pinv(Ac), Bc)))
    else:
        raise ValueError("Matrix Bc must be a 2D NumPy ndarray.")

    return discrete_matrices