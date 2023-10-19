"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 25
"""


import numpy
import scipy

from systemID.core.algorithms.time_varying_markov_parameters_from_time_varying_observer_markov_parameters import time_varying_markov_parameters_from_time_varying_observer_markov_parameters
from systemID.core.algorithms.time_varying_observer_gain_markov_parameters_from_time_varying_observer_markov_parameters import time_varying_observer_gain_markov_parameters_from_time_varying_observer_markov_parameters


def time_varying_observer_kalman_identification_algorithm_with_observer(input_data: numpy.ndarray,
                                                                        output_data: numpy.ndarray,
                                                                        observer_order: int):
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

    # Results
    results = {}

    # Dimensions
    (input_dimension, number_steps, number_experiments) = input_data.shape
    output_dimension = output_data.shape[0]


    # Observer order
    observer_order = min(observer_order, number_steps)


    # Time Varying hki_observer1, hki_observer2 and D matrices
    hki_observer1 = numpy.zeros([(number_steps - 1) * output_dimension, observer_order * input_dimension])
    hki_observer2 = numpy.zeros([(number_steps - 1) * output_dimension, observer_order * output_dimension])
    D = numpy.zeros([output_dimension, input_dimension, number_steps])


    # TVOKID
    for k in range(number_steps):

        # Initialize matrices y and V
        if k == 0:
            number_rows_V = input_dimension
        else:
            number_rows_V = input_dimension + min(observer_order, k) * (input_dimension + output_dimension)
        number_columns_V = number_experiments

        V = numpy.zeros([number_rows_V, number_columns_V])
        y = numpy.zeros([output_dimension, number_columns_V])

        # Populate matrices y and V
        for j in range(number_columns_V):
            y[:, j] = output_data[:, k, j]
            V[0:input_dimension, j] = input_data[:, k, j]
            for i in range(min(observer_order, k)):
                V[input_dimension + i * (input_dimension + output_dimension):input_dimension + (i + 1) * (input_dimension + output_dimension), j] = numpy.concatenate((input_data[:, k - i - 1, j], output_data[:, k - i - 1, j]))

        # Least-Squares solution for Observer Markov Parameters
        Mk = numpy.matmul(y, scipy.linalg.pinv(V))

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
    tvmp = time_varying_markov_parameters_from_time_varying_observer_markov_parameters(D, hki_observer1, hki_observer2, observer_order)

    # Get TV Observer Gain Markov Parameters from TV Observer Markov Parameters
    tvogmp = time_varying_observer_gain_markov_parameters_from_time_varying_observer_markov_parameters(hki_observer2, observer_order)

    results['D'] = D
    results['hki'] = tvmp['hki']
    results['hkio'] = tvogmp['hkio']
    results['hki_observer1'] = hki_observer1
    results['hki_observer2'] = hki_observer2
    results['Error TVOKID'] = scipy.linalg.norm(y - numpy.matmul(Mk, V))

    return results
