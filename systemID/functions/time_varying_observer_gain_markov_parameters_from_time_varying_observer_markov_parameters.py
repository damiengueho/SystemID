"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""



import numpy as np
import scipy.linalg as LA


def time_varying_observer_gain_markov_parameters_from_time_varying_observer_markov_parameters(hki_observer2, observer_order):
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

    # Dimension and number of steps
    output_dimension = int(hki_observer2.shape[1] / observer_order)
    number_steps = int(hki_observer2.shape[0] / output_dimension) + 1

    # Build matrix h2
    h2 = np.eye((number_steps - 1) * output_dimension)
    for i in range(1, number_steps - 1):
        for j in range(max(0, i - observer_order), i):
            h2[i * output_dimension:(i+1) * output_dimension, j * output_dimension:(j+1) * output_dimension] = hki_observer2[i * output_dimension:(i+1) * output_dimension, (i-j-1) * output_dimension:(i-j) * output_dimension]

    # Build matrix r2
    r2 = np.zeros([(number_steps - 1) * output_dimension, (number_steps - 1) * output_dimension])
    for i in range(number_steps - 1):
        for j in range(max(0, i - observer_order + 1), i + 1):
            r2[i * output_dimension:(i + 1) * output_dimension, j * output_dimension:(j + 1) * output_dimension] = hki_observer2[i * output_dimension:(i + 1) * output_dimension, (i - j) * output_dimension:(i - j + 1) * output_dimension]

    # Calculate Markov Parameters
    hkio = np.matmul(LA.inv(h2), r2)

    return hkio
