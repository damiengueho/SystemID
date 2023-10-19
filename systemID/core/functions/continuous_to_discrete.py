"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 25
"""


import numpy
import scipy


def continuous_to_discrete_time_invariant_matrices(dt: float,
                                                   Ac: numpy.ndarray,
                                                   Bc: numpy.ndarray = None,
                                                   integral: bool = None):
    """
        Purpose:
            Provides the discrete time-invariant state matrices given the continuous time-invariant state matrices of
            a linear system.

        Parameters:
            - **dt** (``float``): the discretization time step.
            - **Ac** (``numpy.ndarray``): the continuous time-invariant state matrix.
            - **Bc** (``numpy.ndarray``, optional): the continuous time-invariant input influence matrix.
            - **integral** (``bool``, optional): if ``True``, :math:`B` is calculated using an integral expression.

        Returns:
            - **discrete_matrices** (``list``): a list containing the discrete time-invariant state matrices.

        Imports:
            - ``import numpy``
            - ``import scipy``

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

    if isinstance(Ac, numpy.ndarray) and Ac.ndim == 2 and Ac.shape[0] == Ac.shape[1]:
        A = scipy.linalg.expm(dt * Ac)
        discrete_matrices.append(A)
    else:
        raise ValueError("Matrix Ac must be a 2D NumPy ndarray.")


    if isinstance(Bc, numpy.ndarray) and Bc.ndim == 2:
        if integral:
            def func(t):
                return scipy.linalg.expm(t * Ac)
            def integrand(t, row, col):
                return func(t)[row, col]
            rows, cols = func(0).shape
            result_matrix = numpy.zeros((rows, cols))
            for row in range(rows):
                for col in range(cols):
                    result, _ = scipy.integrate.quad(integrand, 0, dt, args=(row, col))
                    result_matrix[row, col] = result
            discrete_matrices.append(numpy.matmul(result_matrix, Bc))
        else:
            discrete_matrices.append(numpy.matmul(A - numpy.eye(A.shape[0]), numpy.matmul(scipy.linalg.pinv(Ac), Bc)))
    else:
        raise ValueError("Bc must be a numpy.ndarray of shape (state_dimension, input_dimension)")

    return discrete_matrices



def continuous_to_discrete_time_varying_matrices(dt: float,
                                                 Ac: numpy.ndarray,
                                                 Bc: numpy.ndarray = None):
    """
        Purpose:
            Provides the discrete time-varying state matrices given the continuous time-varying state matrices of
            a linear system.

        Parameters:
            - **dt** (``float``): the discretization time step.
            - **Ac** (``numpy.ndarray``): the continuous time-varying state matrix.
            - **Bc** (``numpy.ndarray``, optional): the continuous time-varying input influence matrix.

        Returns:
            - **discrete_matrices** (``list``): a list containing the discrete time-varying state matrices.

        Imports:
            - ``import numpy``
            - ``import scipy``

        Description:
            Given continuous time-varying matrices of a linear system and a step size :math:`\Delta t`, matrix
            differential equations are given by

            .. math::
                :nowrap:

                    \\begin{align}
                        \dot{\Phi}(t, t_k) = A(t)\Phi(t, t_k), \quad \dot{\Psi}(t, t_k) = A(t)\Psi(t, t_k) + I,
                    \\end{align}

            :math:`\forall \ t \in [t_k, t_{k+1}]`, with initial conditions

            .. math::
                :nowrap:

                    \\begin{align}
                        \Phi(t_k, t_k) = \begin{bmatrix} 1 & 0\\ 0 & 1 \end{bmatrix}, \quad
                        \Psi(t_k, t_k) = \begin{bmatrix} 0 & 0\\ 0 & 0 \end{bmatrix}
                    \\end{align}

            such that

            .. math::
                :nowrap:

                    \\begin{align}
                        A_k = \Phi(t_{k+1}, t_k), \quad B_k = \Psi(t_{k+1}, t_k)B,
                    \\end{align}

            would represent the equivalent discrete-time varying system matrices: the state transition matrix
            (equivalent :math:`A_k`) and the convolution integrals (equivalent :math:`B_k` with a zero order hold
            assumption on the inputs).


    """

    discrete_matrices = []

    state_dimension = Ac(0).shape[0]

    def dPhi(Phi, t):
        return numpy.matmul(Ac(t), Phi.reshape(state_dimension, state_dimension)).reshape(state_dimension ** 2)

    def A(tk):
        A = scipy.integrate.odeint(dPhi, numpy.eye(state_dimension).reshape(state_dimension ** 2), numpy.array([tk, tk + dt]),
                   rtol=1e-13, atol=1e-13)
        return A[-1, :].reshape(state_dimension, state_dimension)

    discrete_matrices.append(A)

    def dPsi(Psi, t):
        return numpy.matmul(Ac(t), Psi.reshape(state_dimension, state_dimension)).reshape(
            state_dimension ** 2) + numpy.eye(state_dimension).reshape(state_dimension ** 2)

    def B(tk):
        B = scipy.integrate.odeint(dPsi, numpy.zeros([state_dimension, state_dimension]).reshape(state_dimension ** 2),
                   numpy.array([tk, tk + dt]), rtol=1e-13, atol=1e-13)
        return numpy.matmul(B[-1, :].reshape(state_dimension, state_dimension), Bc(tk))

    discrete_matrices.append(B)

    return discrete_matrices
