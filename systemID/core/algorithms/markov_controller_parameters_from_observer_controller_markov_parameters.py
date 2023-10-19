"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
Date: April 2022
Python: 3.7.7
"""


import numpy


def markov_controller_parameters_from_observer_controller_markov_parameters(observer_controller_markov_parameters: list,
                                                                            number_of_parameters: int = None):
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

    # Return dictionary with results
    results = {}

    # Dimensions
    input_dimension = observer_controller_markov_parameters[0].shape[1]
    output_dimension = observer_controller_markov_parameters[0].shape[0] - input_dimension

    # Get D
    markov_controller_parameters = [observer_controller_markov_parameters[0][0:output_dimension, :]]

    # Number of parameters
    number_observer_controller_markov_parameters = len(observer_controller_markov_parameters)
    # number_of_parameters = max(number_of_parameters, number_observer_controller_markov_parameters)

    # Extract hoc_11, hoc_12, hoc_21 and hoc_22
    hk_11 = ['NaN']
    hk_12 = ['NaN']
    hk_21 = ['NaN']
    hk_22 = ['NaN']
    for i in range(1, min(number_observer_controller_markov_parameters, number_of_parameters)):
        hk_11.append(observer_controller_markov_parameters[i][0:output_dimension, 0:input_dimension])
        hk_12.append(-observer_controller_markov_parameters[i][0:output_dimension, input_dimension:output_dimension + input_dimension])
        hk_21.append(-observer_controller_markov_parameters[i][output_dimension:output_dimension + input_dimension, 0:input_dimension])
        hk_22.append(observer_controller_markov_parameters[i][output_dimension:output_dimension + input_dimension, input_dimension:output_dimension + input_dimension])

    # Get markov_controller_parameters
    for i in range(1, number_of_parameters):
        if i < number_observer_controller_markov_parameters:
            hoc_11 = hk_11[i]
            hoc_12 = hk_12[i]
            hoc_21 = hk_21[i]
            hoc_22 = hk_22[i]
            for j in range(1, i + 1):
                hoc_11 = hoc_11 - numpy.matmul(hk_12[j], markov_controller_parameters[i - j][0:output_dimension, 0:input_dimension])
                hoc_21 = hoc_21 - numpy.matmul(hk_22[j], markov_controller_parameters[i - j][0:output_dimension, 0:input_dimension])
            for j in range(1, i):
                hoc_12 = hoc_12 - numpy.matmul(hk_12[j], markov_controller_parameters[i - j][0:output_dimension, input_dimension:input_dimension+output_dimension])
                hoc_22 = hoc_22 - numpy.matmul(hk_22[j], markov_controller_parameters[i - j][0:output_dimension, input_dimension:input_dimension+output_dimension])
            l1 = numpy.concatenate((hoc_11, hoc_12), axis=1)
            l2 = numpy.concatenate((hoc_21, hoc_22), axis=1)
            hoc = numpy.concatenate((l1, l2), axis=0)
            markov_controller_parameters.append(hoc)
        else:
            hoc_11 = numpy.zeros([output_dimension, input_dimension])
            hoc_12 = numpy.zeros([output_dimension, output_dimension])
            hoc_21 = numpy.zeros([input_dimension, input_dimension])
            hoc_22 = numpy.zeros([input_dimension, output_dimension])
            for j in range(1, number_observer_controller_markov_parameters):
                hoc_11 = hoc_11 - numpy.matmul(hk_12[j], markov_controller_parameters[i - j][0:output_dimension, 0:input_dimension])
                hoc_12 = hoc_12 - numpy.matmul(hk_12[j], markov_controller_parameters[i - j][0:output_dimension, input_dimension:input_dimension+output_dimension])
                hoc_21 = hoc_21 - numpy.matmul(hk_22[j], markov_controller_parameters[i - j][0:output_dimension, 0:input_dimension])
                hoc_22 = hoc_22 - numpy.matmul(hk_22[j], markov_controller_parameters[i - j][0:output_dimension, input_dimension:input_dimension+output_dimension])
            l1 = numpy.concatenate((hoc_11, hoc_12), axis=1)
            l2 = numpy.concatenate((hoc_21, hoc_22), axis=1)
            hoc = numpy.concatenate((l1, l2), axis=0)
            markov_controller_parameters.append(hoc)

    results['markov_controller_parameters'] = markov_controller_parameters

    return results
