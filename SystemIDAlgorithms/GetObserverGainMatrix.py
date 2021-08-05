"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 12
Date: August 2021
Python: 3.7.7
"""


import numpy as np
import scipy.linalg as LA

from SystemIDAlgorithms.GetObservabilityMatrix import getObservabilityMatrix


def getObserverGainMatrix(A, C, observer_gain_markov_parameters, tk, dt, number_steps):

    # Dimensions
    output_dimension, _ = C(tk).shape

    # Compute observability matrix
    O = getObservabilityMatrix(A, C, number_steps, tk, dt)

    # Compute matrix of observer gain markov parameters
    Yo = np.zeros([output_dimension * number_steps, output_dimension])
    for i in range(number_steps):
        Yo[i * output_dimension:(i + 1) * output_dimension, :] = observer_gain_markov_parameters[tk + i + 1]

    # Observer Gain matrix
    G = np.matmul(LA.pinv(O), Yo)

    return G, O, Yo