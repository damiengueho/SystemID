"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 10
Date: April 2021
Python: 3.7.7
"""



import numpy as np

from SystemIDAlgorithms.TimeVaryingObserverKalmanIdentificationAlgorithmObserver import timeVaryingObserverKalmanIdentificationAlgorithmObserver
from SystemIDAlgorithms.GetTVMarkovParametersFromTVObserverMarkovParameters import getTVMarkovParametersFromTVObserverMarkovParameters


def getTimeVaryingHankelMatrix(forced_experiments, free_decay_experiments, p, q, deadbeat_order):

    # Dimensions
    input_dimension = free_decay_experiments.input_dimension
    output_dimension = free_decay_experiments.output_dimension
    number_free_decay_experiments = free_decay_experiments.number_experiments
    number_steps = free_decay_experiments.output_signals[0].number_steps

    # Inputs and outputs
    free_decay_outputs = free_decay_experiments.output_signals

    # Time Varying Y, hki_observer1, hki_observer2 and D matrices
    Y = np.zeros([(p + 1) * output_dimension, number_free_decay_experiments, q])
    hki_observer1 = np.zeros([(number_steps - 1) * output_dimension, (number_steps - 1) * input_dimension])
    hki_observer2 = np.zeros([(number_steps - 1) * output_dimension, (number_steps - 1) * output_dimension])
    D = np.zeros([output_dimension, input_dimension, number_steps])

    # Populate Y matrix
    for k in range(q):
        for i in range(p + 1):
            for j in range(number_free_decay_experiments):
                Y[i * output_dimension:(i + 1) * output_dimension, j, k] = free_decay_outputs[j].data[:, i + k]

    # TVOKID
    for k in range(number_steps):
        Mk, V = timeVaryingObserverKalmanIdentificationAlgorithmObserver(forced_experiments, p + 1, q, deadbeat_order, k)
        D[:, :, k] = Mk[:, 0:input_dimension]
        for j in range(min(max(p + 1 + q - 1, deadbeat_order), k)):
            h_observer = Mk[:, input_dimension + j * (input_dimension + output_dimension):input_dimension + (j + 1) * (input_dimension + output_dimension)]
            h1 = h_observer[:, 0:input_dimension]
            h2 = - h_observer[:, input_dimension:input_dimension + output_dimension]
            hki_observer1[(k - 1) * output_dimension:k * output_dimension, j * input_dimension:(j + 1) * input_dimension] = h1
            hki_observer2[(k - 1) * output_dimension:k * output_dimension, j * output_dimension:(j + 1) * output_dimension] = h2

    # Get Markov Parameters from Observer Markov Parameters
    hki = getTVMarkovParametersFromTVObserverMarkovParameters(D, hki_observer1, hki_observer2, p + 1, q)

    return Y, hki, D, hki_observer1, hki_observer2




