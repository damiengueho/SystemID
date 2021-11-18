"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


import numpy as np
from numpy import linalg as LA


def observerKalmanIdentificationAlgorithmObserver(input_signal, output_signal, observer_order):

    # Get data from Signals
    y = output_signal.data
    u = input_signal.data

    # Get dimensions
    input_dimension = input_signal.dimension
    output_dimension = output_signal.dimension
    number_steps = output_signal.number_steps

    # Build matrix U
    U = np.zeros([(input_dimension + output_dimension) * observer_order + input_dimension, number_steps])
    U[0 * input_dimension:(0 + 1) * input_dimension, :] = u
    for i in range(0, observer_order):
        U[i * (input_dimension + output_dimension) + input_dimension:(i + 1) * (input_dimension + output_dimension) + input_dimension, (i + 1):number_steps] = np.concatenate((u[:, 0:number_steps - 1 - i], y[:, 0:number_steps - 1 - i]), axis=0)

    # Get Y
    Y = np.matmul(y, LA.pinv(U))

    # Get observer Markov parameters
    observer_markov_parameters = [Y[:, 0:input_dimension]]
    for i in range(observer_order):
        observer_markov_parameters.append(Y[:, i * (input_dimension + output_dimension) + input_dimension:(i + 1) * (input_dimension + output_dimension) + input_dimension])

    return observer_markov_parameters, y, U
