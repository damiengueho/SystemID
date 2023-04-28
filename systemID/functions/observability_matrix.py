"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
Date: April 2022
Python: 3.7.7
"""


import numpy as np


def observability_matrix(A, C, number_steps, **kwargs):
    """
        Purpose:
            This function calculates the observability matrix at time **tk**.

        Parameters:
            - **A** (``fun``): the system matrix.
            - **C** (``fun``): the output influence matrix.
            - **number_steps** (``int``): the block size :math:`p` of the observability matrix.
            - **tk** (``float``, *optional*): the starting time step. If not specified, its value is :math:`0`.
            - **dt** (``float``, *optional*): the length of the time step for which the zero-order hold approximation for the system matrices is valid. If not specified, its value is :math:`0`.

        Returns:
            - **O** (``ndarray``): the corresponding observability matrix.

        Imports:
            - ``import numpy as np``

        Description:
            The algorithm builds the observability matrix :math:`O_k^{(p)}` from time step **tk** such that

            .. math::

                O_k^{(p)} = \\begin{bmatrix} C_k \\\\ C_{k+1}A_k \\\\ C_{k+2}A_{k+1}A_k \\\\ \\vdots \\\\ C_{k+p-1}A_{k+p-2}\\cdots A_k \\end{bmatrix}.

        See Also:
            -
    """

    tk = kwargs.get('tk', 0)
    dt = kwargs.get('dt', 0)

    (output_dimension, state_dimension) = C(tk).shape

    O = np.zeros([number_steps * output_dimension, state_dimension])

    if number_steps <= 0:
        return np.zeros([number_steps * output_dimension, state_dimension])
    if number_steps == 1:
        O[0:output_dimension, :] = C(tk)
        return O
    if number_steps > 1:
        O[0:output_dimension, :] = C(tk)
        for j in range(1, number_steps):
            temp = A(tk)
            for i in range(0, j - 1):
                temp = np.matmul(A(tk + (i + 1) * dt), temp)
            O[j * output_dimension:(j + 1) * output_dimension, :] = np.matmul(C(tk + j * dt), temp)
        return O
