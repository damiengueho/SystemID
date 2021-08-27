"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 15
Date: August 2021
Python: 3.7.7
"""


import numpy as np
from numpy import linalg as LA


def observerKalmanIdentificationAlgorithmObserverWithInitialCondition(input_signal, output_signal, deadbeat_order):

    # Get data from Signals
    y = output_signal.data
    u = input_signal.data

    # Get dimensions
    input_dimension = input_signal.dimension
    output_dimension = output_signal.dimension
    number_steps = output_signal.number_steps

    # Build matrix U
    U = np.zeros([(input_dimension + output_dimension) * deadbeat_order + input_dimension, number_steps - deadbeat_order])
    U[0 * input_dimension:(0 + 1) * input_dimension, :] = u[:, deadbeat_order:number_steps]
    for i in range(0, deadbeat_order):
        U[i * (input_dimension + output_dimension) + input_dimension:(i + 1) * (input_dimension + output_dimension) + input_dimension, :] = np.concatenate((u[:, deadbeat_order - 1 - i:number_steps - 1 - i], y[:, deadbeat_order - 1 - i:number_steps - 1 - i]), axis=0)

    # Get Y
    Y = np.matmul(y[:, deadbeat_order:number_steps], LA.pinv(U))

    # Get observer Markov parameters
    observer_markov_parameters = [Y[:, 0:input_dimension]]
    for i in range(deadbeat_order):
        observer_markov_parameters.append(Y[:, i * (input_dimension + output_dimension) + input_dimension:(i + 1) * (input_dimension + output_dimension) + input_dimension])

    return observer_markov_parameters, y, U
