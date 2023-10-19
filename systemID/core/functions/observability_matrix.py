"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 25
"""


import numpy
from typing import Callable

def observability_matrix(A: Callable[[float], numpy.ndarray],
                         C: Callable[[float], numpy.ndarray],
                         size: int = None,
                         tk: float = 0,
                         dt: float = 0):
    """
        Purpose:
            This function returns the observability matrix at time **tk**.

        Parameters:
            - **A** (``func: (float) -> numpy.ndarray``): the system matrix.
            - **C** (``func: (float) -> numpy.ndarray``): the output influence matrix.
            - **size** (``int``, *optional*): the number of blocks :math:`p` of the observability matrix.
            - **tk** (``float``, *optional*): the starting time step. If not specified, its value is :math:`0`.
            - **dt** (``float``, *optional*): the length of the time step for which the zero-order hold approximation
             for the system matrices is valid. If not specified, its value is :math:`0`.

        Returns:
            - **O** (``numpy.ndarray``): the corresponding observability matrix.

        Imports:
            - ``import numpy``

        Description:
            The algorithm builds the observability matrix :math:`O_k^{(p)}` from time step **tk** such that

            .. math::

                O_k^{(p)} = \\begin{bmatrix} C_k \\\\ C_{k+1}A_k \\\\ C_{k+2}A_{k+1}A_k \\\\ \\vdots \\\\ C_{k+p-1}A_{k+p-2}\\cdots A_k \\end{bmatrix}.

        See Also:
            -
    """

    (output_dimension, state_dimension) = C(tk).shape

    O = numpy.zeros([size * output_dimension, state_dimension])

    if size <= 0:
        return numpy.zeros([size * output_dimension, state_dimension])
    if size == 1:
        O[0:output_dimension, :] = C(tk)
        return O
    if size > 1:
        O[0:output_dimension, :] = C(tk)
        for j in range(1, size):
            temp = A(tk)
            for i in range(0, j - 1):
                temp = numpy.matmul(A(tk + (i + 1) * dt), temp)
            O[j * output_dimension:(j + 1) * output_dimension, :] = numpy.matmul(C(tk + j * dt), temp)
        return O
