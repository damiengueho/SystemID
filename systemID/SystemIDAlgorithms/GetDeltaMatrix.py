"""
Author: Damien GUEHO
Copyright: Copyright (C) 2022 Damien GUEHO
License: Public Domain
Version: 23
Date: April 2022
Python: 3.7.7
"""


import numpy as np
from systemID.SystemIDAlgorithms.GetTimeVaryingMarkovParameters import getTimeVaryingMarkovParameters


def getDeltaMatrix(A, B, C, D, tk, dt, number_steps):
    """
        Purpose:
            This function creates the :math:`\Delta_k` matrix defined below.

        Parameters:
            - **A** (``fun``): the system matrix.
            - **B** (``fun``): the input influence matrix.
            - **C** (``fun``): the output influence matrix.
            - **D** (``fun``): the direct transmission matrix.
            - **tk** (``float``): the time step from which the :math:`\Delta_k` matrix is built.
            - **dt** (``float``): the length of the time step for which the zero-order hold approximation for the system matrices is valid.
            - **number_steps** (``int``): the number of steps :math:`p` for which the matrix will be built.

        Returns:
            - **Delta_k** (``ndarray``): the corresponding :math:`\Delta_k` matrix.

        Imports:
            - ``import numpy as np``
            - ``from systemID.SystemIDAlgorithms.GetTimeVaryingMarkovParameters import getTimeVaryingMarkovParameters``

        Description:
            Returns the matrix

            .. math::

                    \Delta_k = \\begin{bmatrix}
                                D_k & & & & \\\\
                                C_{k+1}B_k & D_{k+1} & & & \\\\
                                C_{k+2}A_{k+1}B_k & C_{k+2}B_{k+1} & D_{k+2} & & \\\\
                                \\vdots & \\vdots & \\vdots & \\ddots & \\\\
                                C_{k+p-1}A_{k+p-2}\\cdots B_k & C_{k+p-1}A_{k+p-2}\\cdots B_{k+1} & C_{k+p-1}A_{k+p-2}\\cdots B_{k+2} & \\cdots & D_{k+p-1}
                               \\end{bmatrix}.


        See Also:
            - :py:mod:`~SystemIDAlgorithms.GetTimeVaryingMarkovParameters.getTimeVaryingMarkovParameters`
    """

    # Get dimensions
    output_dimension, input_dimension = D(tk).shape

    # Get Delta Matrix
    Delta = np.zeros([number_steps * output_dimension, number_steps * input_dimension])
    for i in range(number_steps):
        time_varying_markov_parameters = getTimeVaryingMarkovParameters(A, B, C, D, i*dt + tk, dt, number_steps=i + 1)
        Delta[i*output_dimension:(i+1)*output_dimension, 0:(i+1)*input_dimension] = np.concatenate(time_varying_markov_parameters[::-1], axis=1)

    return Delta

