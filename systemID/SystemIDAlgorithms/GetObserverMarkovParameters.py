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


def getObserverMarkovParameters(A, B, C, D, G, number_steps):
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

    # Dimensions
    output_dimension, input_dimension = D(0).shape
    state_dimension, _ = A(0).shape

    # Initialisation
    observer_markov_parameters = [D(0)]
    B_bar = np.zeros([state_dimension, output_dimension + input_dimension])
    B_bar[:, 0:input_dimension] = B(0) + np.matmul(G(0), D(0))
    B_bar[:, input_dimension:output_dimension + input_dimension] = -G(0)
    A_bar = A(0) + np.matmul(G(0), C(0))

    for i in range(number_steps - 1):
        observer_markov_parameters.append(np.matmul(C(0), np.matmul(matpow(A_bar, i), B_bar)))

    return observer_markov_parameters
