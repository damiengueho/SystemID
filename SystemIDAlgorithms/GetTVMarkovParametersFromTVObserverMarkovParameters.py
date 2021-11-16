"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""



import numpy as np
import scipy.linalg as LA


def getTVMarkovParametersFromTVObserverMarkovParameters(D, hki_observer1, hki_observer2, observer_order):

    # Dimensions and number of steps
    output_dimension, input_dimension, number_steps = D.shape

    # Build matrix h2
    h2 = np.eye((number_steps - 1) * output_dimension)
    for i in range(1, number_steps - 1):
        for j in range(max(0, i - observer_order), i):
            h2[i * output_dimension:(i+1) * output_dimension, j * output_dimension:(j+1) * output_dimension] = hki_observer2[i * output_dimension:(i+1) * output_dimension, (i-j-1) * output_dimension:(i-j) * output_dimension]

    # Build matrix r
    r = np.zeros([(number_steps - 1) * output_dimension, (number_steps - 1) * input_dimension])
    for i in range(number_steps - 1):
        for j in range(max(0, i - observer_order + 1), i + 1):
            r[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension] = hki_observer1[i * output_dimension:(i + 1) * output_dimension, (i - j) * input_dimension:(i - j + 1) * input_dimension] \
                                                                                                                - np.matmul(hki_observer2[i * output_dimension:(i + 1) * output_dimension, (i - j) * output_dimension:(i - j + 1) * output_dimension], D[:, :, j])

    # Calculate Markov Parameters
    hki = np.matmul(LA.inv(h2), r)

    return hki, h2, r
