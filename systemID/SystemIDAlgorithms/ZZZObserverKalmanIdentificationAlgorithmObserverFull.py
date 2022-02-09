"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""


import numpy as np
from numpy import linalg as LA


def observerKalmanIdentificationAlgorithmObserverFull(input_signal, output_signal):

    # Get data from Signals
    y = output_signal.data
    u = input_signal.data

    # Get dimensions
    input_dimension = input_signal.dimension
    output_dimension = output_signal.dimension
    number_steps = output_signal.number_steps

    # Build matrix U
    U = np.zeros([(input_dimension + output_dimension) * number_steps + input_dimension, number_steps])
    U[0 * input_dimension:(0 + 1) * input_dimension, 0:number_steps] = u[:, 0:number_steps - 0]
    for i in range(0, number_steps - 1):
        U[i * (input_dimension + output_dimension) + input_dimension:(i + 1) * (input_dimension + output_dimension) + input_dimension, (i + 1):number_steps] = np.concatenate((u[:, 0:number_steps - 1 - i], y[:, 0:number_steps - 1 - i]), axis=0)

    # Get Y
    Y = np.matmul(y, LA.pinv(U))

    # Get observer Markov parameters
    observer_markov_parameters = [Y[:, 0:input_dimension]]
    for i in range(number_steps - 1):
        observer_markov_parameters.append(Y[:, i * (input_dimension + output_dimension) + input_dimension:(i + 1) * (input_dimension + output_dimension) + input_dimension])

    return observer_markov_parameters
