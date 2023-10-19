"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 25
"""


import numpy
import scipy


def time_varying_markov_parameters_from_time_varying_observer_markov_parameters(D: numpy.ndarray,
                                                                                hki_observer1: list,
                                                                                hki_observer2: list,
                                                                                observer_order: int
):
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

    # Dimensions and number of steps
    output_dimension, input_dimension, number_steps = D.shape

    # Build matrix h2
    h2 = numpy.eye((number_steps - 1) * output_dimension)
    for i in range(1, number_steps - 1):
        for j in range(max(0, i - observer_order), i):
            h2[i * output_dimension:(i+1) * output_dimension, j * output_dimension:(j+1) * output_dimension] = hki_observer2[i * output_dimension:(i+1) * output_dimension, (i-j-1) * output_dimension:(i-j) * output_dimension]

    # Build matrix r
    r = numpy.zeros([(number_steps - 1) * output_dimension, (number_steps - 1) * input_dimension])
    for i in range(number_steps - 1):
        for j in range(max(0, i - observer_order + 1), i + 1):
            r[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension] = hki_observer1[i * output_dimension:(i + 1) * output_dimension, (i - j) * input_dimension:(i - j + 1) * input_dimension] \
                                                                                                                - numpy.matmul(hki_observer2[i * output_dimension:(i + 1) * output_dimension, (i - j) * output_dimension:(i - j + 1) * output_dimension], D[:, :, j])

    # Calculate Markov Parameters
    hki = numpy.matmul(scipy.linalg.inv(h2), r)

    results['hki'] = hki
    results['h2'] = h2
    results['r'] = r

    return results
