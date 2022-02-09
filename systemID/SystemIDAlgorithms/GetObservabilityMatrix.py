"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""


import numpy as np


def getObservabilityMatrix(A, C, number_steps, tk, dt):
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

    (state_dimension, _) = A(tk).shape
    (output_dimension, _) = C(tk).shape

    O = np.zeros([number_steps * output_dimension, state_dimension])

    O[0:output_dimension, :] = C(tk)

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
