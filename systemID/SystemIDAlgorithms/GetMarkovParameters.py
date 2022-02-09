"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""


import numpy as np
from scipy.linalg import fractional_matrix_power as matpow


def getMarkovParameters(A, B, C, D, number_steps):
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

    markov_parameters = [D(0)]

    for i in range(number_steps - 1):
        markov_parameters.append(np.matmul(C(0), np.matmul(matpow(A(0), i), B(0))))

    return markov_parameters
