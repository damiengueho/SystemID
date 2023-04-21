"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""



import numpy as np
import scipy.linalg as LA

from systemID.functions.time_varying_markov_parameters_from_time_varying_observer_markov_parameters import time_varying_markov_parameters_from_time_varying_observer_markov_parameters
from systemID.functions.time_varying_observer_gain_markov_parameters_from_time_varying_observer_markov_parameters import time_varying_observer_gain_markov_parameters_from_time_varying_observer_markov_parameters


def time_varying_observer_kalman_identification_algorithm_with_observer(input_signals, output_signals, **kwargs):
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
    input_dimension = input_signals[0].dimension
    output_dimension = output_signals[0].dimension
    number_steps = output_signals[0].number_steps
    number_experiments = len(output_signals)

    # Observer order
    observer_order = kwargs.get("observer_order", number_steps)


    # Time Varying hki_observer1, hki_observer2 and D matrices
    hki_observer1 = np.zeros([(number_steps - 1) * output_dimension, observer_order * input_dimension])
    hki_observer2 = np.zeros([(number_steps - 1) * output_dimension, observer_order * output_dimension])
    D = np.zeros([output_dimension, input_dimension, number_steps])


    # TVOKID
    for k in range(number_steps):

        # Initialize matrices y and V
        if k == 0:
            number_rows_V = input_dimension
        else:
            number_rows_V = input_dimension + min(observer_order, k) * (input_dimension + output_dimension)
        number_columns_V = number_experiments

        V = np.zeros([number_rows_V, number_columns_V])
        y = np.zeros([output_dimension, number_columns_V])

        # Populate matrices y and V
        for j in range(number_columns_V):
            y[:, j] = output_signals[j].data[:, k]
            V[0:input_dimension, j] = input_signals[j].data[:, k]
            for i in range(min(observer_order, k)):
                V[input_dimension + i * (input_dimension + output_dimension):input_dimension + (i + 1) * (input_dimension + output_dimension), j] = np.concatenate((input_signals[j].data[:, k - i - 1], output_signals[j].data[:, k - i - 1]))

        # Least-Squares solution for Observer Markov Parameters
        Mk = np.matmul(y, LA.pinv(V))
        # print('Error TVOKID', LA.norm(y - np.matmul(Mk, V)))

        # Extract Dk
        D[:, :, k] = Mk[:, 0:input_dimension]

        # Extract Observer Markov Parameters
        for j in range(min(observer_order, k)):
            h_observer = Mk[:, input_dimension + j * (input_dimension + output_dimension):input_dimension + (j + 1) * (input_dimension + output_dimension)]
            h1 = h_observer[:, 0:input_dimension]
            h2 = - h_observer[:, input_dimension:input_dimension + output_dimension]
            hki_observer1[(k - 1) * output_dimension:k * output_dimension, j * input_dimension:(j + 1) * input_dimension] = h1
            hki_observer2[(k - 1) * output_dimension:k * output_dimension, j * output_dimension:(j + 1) * output_dimension] = h2

    # Get TV Markov Parameters from TV Observer Markov Parameters
    hki, h2, r = time_varying_markov_parameters_from_time_varying_observer_markov_parameters(D, hki_observer1, hki_observer2, observer_order)

    # Get TV Observer Gain Markov Parameters from TV Observer Markov Parameters
    hkio = time_varying_observer_gain_markov_parameters_from_time_varying_observer_markov_parameters(hki_observer2, observer_order)

    return D, hki, hkio, hki_observer1, hki_observer2
