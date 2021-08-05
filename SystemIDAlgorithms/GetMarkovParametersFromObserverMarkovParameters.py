"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 12
Date: August 2021
Python: 3.7.7
"""


import numpy as np


def getMarkovParametersFromObserverMarkovParameters(observer_markov_parameters, number_to_calculate):

    # Dimensions
    output_dimension, input_dimension = observer_markov_parameters[0].shape

    # Get D
    markov_parameters = [observer_markov_parameters[0]]

    # Number of Observer Markov parameters
    number_observer_markov_parameters = len(observer_markov_parameters)

    # Extract hk1 and hk2
    hk1 = ['NaN']
    hk2 = ['NaN']
    for i in range(1, min(number_observer_markov_parameters, number_to_calculate)):
        hk1.append(observer_markov_parameters[i][:, 0:input_dimension])
        hk2.append(-observer_markov_parameters[i][:, input_dimension:output_dimension + input_dimension])

    # Get hk
    for i in range(1, number_to_calculate):
        if i < number_observer_markov_parameters:
            hk = hk1[i]
            for j in range(1, i + 1):
                hk = hk - np.matmul(hk2[j], markov_parameters[i - j])
            markov_parameters.append(hk)
        else:
            hk = np.zeros([output_dimension, input_dimension])
            for j in range(1, number_observer_markov_parameters):
                hk = hk - np.matmul(hk2[j], markov_parameters[i - j])
            markov_parameters.append(hk)

    return markov_parameters
