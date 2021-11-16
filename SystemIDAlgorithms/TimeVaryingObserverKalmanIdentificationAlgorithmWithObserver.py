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

from SystemIDAlgorithms.GetTVMarkovParametersFromTVObserverMarkovParameters import getTVMarkovParametersFromTVObserverMarkovParameters
from SystemIDAlgorithms.GetTVObserverGainMarkovParametersFromTVObserverMarkovParameters import getTVObserverGainMarkovParametersFromTVObserverMarkovParameters


def timeVaryingObserverKalmanIdentificationAlgorithmWithObserver(forced_experiments, **kwargs):

    # Dimensions
    input_dimension = forced_experiments.input_dimension
    output_dimension = forced_experiments.output_dimension
    number_steps = forced_experiments.output_signals[0].number_steps
    number_forced_experiments = forced_experiments.number_experiments

    # Observer order
    observer_order = kwargs.get("observer_order", number_steps)

    # Inputs and outputs
    forced_inputs = forced_experiments.input_signals
    forced_outputs = forced_experiments.output_signals


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
        number_columns_V = number_forced_experiments

        V = np.zeros([number_rows_V, number_columns_V])
        y = np.zeros([output_dimension, number_columns_V])

        # Populate matrices y and V
        for j in range(number_columns_V):
            y[:, j] = forced_outputs[j].data[:, k]
            V[0:input_dimension, j] = forced_inputs[j].data[:, k]
            for i in range(min(observer_order, k)):
                V[input_dimension + i * (input_dimension + output_dimension):input_dimension + (i + 1) * (input_dimension + output_dimension), j] = np.concatenate((forced_inputs[j].data[:, k - i - 1], forced_outputs[j].data[:, k - i - 1]))

        # Least-Squares solution for Observer Markov Parameters
        Mk = np.matmul(y, LA.pinv(V))

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
    hki, h2, r = getTVMarkovParametersFromTVObserverMarkovParameters(D, hki_observer1, hki_observer2, observer_order)

    # Get TV Observer Gain Markov Parameters from TV Observer Markov Parameters
    hkio = getTVObserverGainMarkovParametersFromTVObserverMarkovParameters(hki_observer2, observer_order)

    return D, hki, hkio




