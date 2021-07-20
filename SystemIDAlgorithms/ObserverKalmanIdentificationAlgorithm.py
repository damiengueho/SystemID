"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 11
Date: July 2021
Python: 3.7.7
"""


import numpy as np
from numpy import linalg as LA


def observerKalmanIdentificationAlgorithm(input_signal, output_signal, deadbeat_order):

    # Get data from Signals
    y = output_signal.data
    u = input_signal.data

    # Get dimensions
    input_dimension = input_signal.dimension
    number_steps = output_signal.number_steps

    # Build matrix U
    U = np.zeros([input_dimension * deadbeat_order, number_steps])
    for i in range(0, deadbeat_order):
        U[i * input_dimension:(i + 1) * input_dimension, i:number_steps] = u[:, 0:number_steps - i]

    # Get Y
    Y = np.matmul(y, LA.pinv(U))

    # Get Markov parameters
    markov_parameters = [Y[:, 0:input_dimension]]
    for i in range(deadbeat_order - 1):
        markov_parameters.append(Y[:, i * input_dimension + input_dimension:(i + 1) * input_dimension + input_dimension])

    return markov_parameters
