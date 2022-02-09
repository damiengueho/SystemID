"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""



import numpy as np
import scipy.linalg as LA


def timeVaryingObserverKalmanIdentificationAlgorithmObserver(forced_experiments, p, q, observer_order, k):

    # Dimensions
    output_dimension = forced_experiments.output_dimension
    input_dimension = forced_experiments.input_dimension

    # Number of experiments
    number_forced_experiments = forced_experiments.number_experiments

    # Inputs and outputs
    forced_inputs = forced_experiments.input_signals
    forced_outputs = forced_experiments.output_signals

    # Build matrices y and V
    if k == 0:
        number_rows_V = input_dimension
    else:
        number_rows_V = input_dimension + min(max(p + q - 1, observer_order), k) * (input_dimension + output_dimension)
    number_columns_V = number_forced_experiments

    V = np.zeros([number_rows_V, number_columns_V])
    y = np.zeros([output_dimension, number_columns_V])
    for j in range(number_columns_V):
        y[:, j] = forced_outputs[j].data[:, k]
        V[0:input_dimension, j] = forced_inputs[j].data[:, k]
        for l in range(min(max(p + q - 1, observer_order), k)):
            V[input_dimension + l * (input_dimension + output_dimension):input_dimension + (l + 1) * (input_dimension + output_dimension), j] = np.concatenate((forced_inputs[j].data[:, k - l - 1], forced_outputs[j].data[:, k - l - 1]))

    # Least-Squares solution for Observer Markov Parameters
    Mk = np.matmul(y, LA.pinv(V))

    return Mk, V
