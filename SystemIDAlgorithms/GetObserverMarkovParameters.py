"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


import numpy as np
from scipy.linalg import fractional_matrix_power as matpow


def getObserverMarkovParameters(A, B, C, D, G, number_steps):

    # Dimensions
    output_dimension, input_dimension = D(0).shape
    state_dimension, _ = A(0).shape

    # Initialisation
    ObserverMarkovParameters = [D(0)]
    B_bar = np.zeros([state_dimension, output_dimension + input_dimension])
    B_bar[:, 0:input_dimension] = B(0) + np.matmul(G(0), D(0))
    B_bar[:, input_dimension:output_dimension + input_dimension] = -G(0)
    A_bar = A(0) + np.matmul(G(0), C(0))

    for i in range(number_steps - 1):
        ObserverMarkovParameters.append(np.matmul(C(0), np.matmul(matpow(A_bar, i), B_bar)))

    return ObserverMarkovParameters
