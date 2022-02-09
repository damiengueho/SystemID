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


def getTimeVaryingObserverGainMatrix(A, C, hkio, order, dt):
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
    output_dimension, _ = C(0).shape
    state_dimension, _ = A(0).shape

    # Number of steps
    number_steps = int(hkio.shape[0] / output_dimension) + 1

    # Initialize Gain matrices
    G_mat = np.zeros([state_dimension, output_dimension, number_steps])

    print(number_steps - 1 - order)

    # Compute Gain matrices
    for k in range(number_steps - 1 - order):
        print(k)
        O = getObservabilityMatrix(A, C, order, (k + 1) * dt, dt)
        Yo = hkio[k * output_dimension:(k + order) * output_dimension, k * output_dimension:(k + 1) * output_dimension]
        G_mat[:, :, k] = np.matmul(LA.pinv(O), Yo)

    def G(tk):
        return G_mat[:, :, int(round(tk / dt))]

    return G
