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


def observerKalmanIdentificationAlgorithmWithObserver(input_signal, output_signal, **kwargs):

    # Get data from Signals
    y = output_signal.data
    u = input_signal.data

    # Get dimensions
    input_dimension = input_signal.dimension
    output_dimension = output_signal.dimension

    # Get number of Markov parameters to compute
    number_steps = output_signal.number_steps
    observer_order = min(kwargs.get('observer_order', number_steps - 1), number_steps - 1)
    stable_order = kwargs.get('stable_order', 0)

    # Build matrix U
    U = np.zeros([(input_dimension + output_dimension) * observer_order + input_dimension, number_steps - stable_order])
    U[0 * input_dimension:(0 + 1) * input_dimension, :] = u[:, stable_order:number_steps]
    for i in range(1, observer_order + 1):
        U[(i - 1) * (input_dimension + output_dimension) + input_dimension:i * (input_dimension + output_dimension) + input_dimension, max(0, i - stable_order):number_steps - stable_order] = np.concatenate((u[:, stable_order - min(i, stable_order):number_steps - i], y[:, stable_order - min(i, stable_order):number_steps - i]), axis=0)

    # Get Y
    Y = np.matmul(y[:, stable_order:], LA.pinv(U))

    # Get observer Markov parameters
    observer_markov_parameters = [Y[:, 0:input_dimension]]
    for i in range(observer_order):
        observer_markov_parameters.append(Y[:, i * (input_dimension + output_dimension) + input_dimension:(i + 1) * (input_dimension + output_dimension) + input_dimension])

    return observer_markov_parameters, y, U
