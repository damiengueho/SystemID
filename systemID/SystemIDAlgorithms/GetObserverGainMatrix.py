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


def getObserverGainMatrix(A, C, observer_gain_markov_parameters, tk, dt, order):
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
    output_dimension, _ = C(tk).shape

    # Compute observability matrix
    O = getObservabilityMatrix(A, C, order, tk, dt)

    # Compute matrix of observer gain markov parameters
    Yo = np.zeros([output_dimension * order, output_dimension])
    for i in range(order):
        Yo[i * output_dimension:(i + 1) * output_dimension, :] = observer_gain_markov_parameters[tk + i + 1]

    # Observer Gain matrix
    G_mat = np.matmul(LA.pinv(O), Yo)

    def G(t):
        return G_mat

    return G, O, Yo