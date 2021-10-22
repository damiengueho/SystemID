"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 18
Date: October 2021
Python: 3.7.7
"""


import numpy as np
from numpy import linalg as LA


def observerKalmanIdentificationAlgorithm(input_signal, output_signal, **kwargs):

    # Get data from Signals
    y = output_signal.data
    u = input_signal.data

    # Get dimensions
    input_dimension = input_signal.dimension

    # Get number of Markov parameters to compute
    number_steps = output_signal.number_steps
    number_of_parameters = min(kwargs.get('number_of_parameters', number_steps), number_steps)
    stable_order = kwargs.get('stable_order', 0)

    # Build matrix U
    U = np.zeros([input_dimension * number_of_parameters, number_steps - stable_order])
    for i in range(0, number_of_parameters):
        U[i * input_dimension:(i + 1) * input_dimension, max(0, i - stable_order):number_steps - stable_order] = u[:, stable_order - min(i, stable_order):number_steps - i]

    # Get Y
    Y = np.matmul(y[:, stable_order:], LA.pinv(U))

    # Get Markov parameters
    markov_parameters = [Y[:, 0:input_dimension]]
    for i in range(number_of_parameters - 1):
        markov_parameters.append(Y[:, i * input_dimension + input_dimension:(i + 1) * input_dimension + input_dimension])

    return markov_parameters, U, y[:, stable_order:]
