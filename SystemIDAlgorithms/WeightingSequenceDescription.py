"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 16
Date: September 2021
Python: 3.7.7
"""


import numpy as np

from ClassesGeneral.ClassSignal import DiscreteSignal


def weightingSequenceDescription(input_signal, markov_parameters, **kwargs):

    # Observer
    observer = kwargs.get('observer', False)

    # Get dimensions
    if observer:
        output_dimension, _ = markov_parameters[0].shape
        input_dimension = input_signal.dimension
    else:
        output_dimension, input_dimension = markov_parameters[0].shape

    # Get number of steps
    number_steps = input_signal.number_steps

    # Get observer order
    observer_order = len(markov_parameters)

    # Get data
    u = input_signal.data

    # Build matrices
    if observer:
        y = np.zeros([output_dimension, number_steps])
        for i in range(number_steps):
            y[:, i] = np.matmul(markov_parameters[0], u[:, i])
            for j in range(1, min(i + 1, observer_order)):
                y[:, i] = y[:, i] + np.matmul(markov_parameters[j], np.concatenate((u[:, i - j], y[:, i - j])))
        output_signal = DiscreteSignal(output_dimension, input_signal.total_time, input_signal.frequency, signal_shape='External', data=y)

    else:
        U = np.zeros([input_dimension * observer_order, min(observer_order, number_steps)])
        for i in range(0, min(observer_order, number_steps)):
            print(i)
            U[i * input_dimension:(i + 1) * input_dimension, i:number_steps] = u[:, 0:number_steps - i]

        # Calculate Y and y
        Y = np.concatenate(markov_parameters, axis=1)
        y = np.matmul(Y, U)

        output_signal = DiscreteSignal(output_dimension, input_signal.total_time, input_signal.frequency, signal_shape='External', data=y)

    return output_signal
