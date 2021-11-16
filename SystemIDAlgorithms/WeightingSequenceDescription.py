"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


import numpy as np

from ClassesGeneral.ClassSignal import DiscreteSignal


def weightingSequenceDescription(input_signal, markov_parameters, **kwargs):
    """
    Purpose:
        Compute the weighting sequence description :math:`\\boldsymbol{y}_k = \displaystyle\sum_{i=0}^ph_i\\boldsymbol{u}_{k-i}` of a linear system using the system markov parameters.

    Parameters:
        - **input_signal** (``DiscreteSignal``): the input signal
        - **markov_parameters** (``list``): list of markov parameters (weights) to use
        - ****kwargs** (``bool``): observer

    Returns:
        - **output_signal** (``DiscreteSignal``): the output signal

    Imports:
        - ``import numpy as np``
        - ``from ClassesGeneral.ClassSignal import DiscreteSignal``

    Description:
        Uses matrix multiplication. Inverse of OKID.

    See Also:
        :mod:`ClassesGeneral.ClassSignal.DiscreteSignal`
    """

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
    number_of_parameters = len(markov_parameters)
    stable_order = kwargs.get('stable_order', 0)

    # Get data
    u = input_signal.data

    # Build matrices
    if observer:
        y_meas = kwargs.get('reference_output_signal', DiscreteSignal(output_dimension, input_signal.total_time, input_signal.frequency)).data
        # Build matrix U
        U = np.zeros([(input_dimension + output_dimension) * min(number_of_parameters - 1, number_steps - 1) + input_dimension, number_steps - stable_order])
        U[0 * input_dimension:(0 + 1) * input_dimension, :] = u[:, stable_order:number_steps]
        for i in range(1, min(number_of_parameters - 1, number_steps - 1) + 1):
            U[(i - 1) * (input_dimension + output_dimension) + input_dimension:i * (input_dimension + output_dimension) + input_dimension, max(0, i - stable_order):number_steps - stable_order] = np.concatenate((u[:, stable_order - min(i, stable_order):number_steps - i], y_meas[:, stable_order - min(i, stable_order):number_steps - i]), axis=0)

        # Calculate Y and y
        Y = np.concatenate(markov_parameters[0:min(number_of_parameters, number_steps - 1) + 1], axis=1)
        y = np.matmul(Y, U)
        y = np.concatenate((np.zeros([output_dimension, stable_order]), y), axis=1)
        y[:, 0:stable_order] = np.NaN

        output_signal = DiscreteSignal(output_dimension, input_signal.total_time, input_signal.frequency, signal_shape='External', data=y)


    else:
        # Build matrix U
        U = np.zeros([input_dimension * min(number_of_parameters, number_steps), number_steps - stable_order])
        for i in range(0, min(number_of_parameters, number_steps)):
            U[i * input_dimension:(i + 1) * input_dimension, max(0, i - stable_order):number_steps - stable_order] = u[:, stable_order - min(i, stable_order):number_steps - i]

        # Calculate Y and y
        Y = np.concatenate(markov_parameters[0:min(number_of_parameters, number_steps)], axis=1)
        y = np.matmul(Y, U)
        y = np.concatenate((np.zeros([output_dimension, stable_order]), y), axis=1)
        y[:, 0:stable_order] = np.NaN

        output_signal = DiscreteSignal(output_dimension, input_signal.total_time, input_signal.frequency, signal_shape='External', data=y)

    return output_signal
