"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 10
Date: April 2021
Python: 3.7.7
"""



import numpy as np
import scipy.linalg as LA


def timeVaryingObserverKalmanIdentificationAlgorithmObserver(forced_experiments, p, q, deadbeat_order, k):

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
        #number_rows_V = input_dimension + min(max(p + q - 1, deadbeat_order), k) * (input_dimension + output_dimension) - output_dimension
        number_rows_V = input_dimension + min(max(p + q - 1, deadbeat_order), k) * (input_dimension + output_dimension)
    number_columns_V = number_forced_experiments


    V = np.zeros([number_rows_V, number_columns_V])
    y = np.zeros([output_dimension, number_columns_V])
    for j in range(number_columns_V):
        y[:, j] = forced_outputs[j].data[:, k]
        V[0:input_dimension, j] = forced_inputs[j].data[:, k]
        for l in range(min(max(p + q - 1, deadbeat_order), k)):
            # if k > 0 and l == min(max(p + q - 1, deadbeat_order), k) - 1:
            #     V[input_dimension + l * (input_dimension + output_dimension):input_dimension + (l + 1) * (input_dimension + output_dimension), j] = forced_inputs[j].data[:, k - l - 1]
            # else:
            #     V[input_dimension + l * (input_dimension + output_dimension):input_dimension + (l + 1) * (input_dimension + output_dimension), j] = np.concatenate((forced_inputs[j].data[:, k - l - 1], forced_outputs[j].data[:, k - l - 1]))
            V[input_dimension + l * (input_dimension + output_dimension):input_dimension + (l + 1) * (input_dimension + output_dimension), j] = np.concatenate((forced_inputs[j].data[:, k - l - 1], forced_outputs[j].data[:, k - l - 1]))

    # print('Shape y:', y.shape)
    #print('Shape V:', V.shape)
    #print('Y', y)
    #print('V', V)
    # Least-Squares solution for Observer Markov Parameters
    Mk = np.matmul(y, LA.pinv(V))
    #Mk = np.matmul(y, np.matmul(V.T, LA.inv(np.matmul(V, V.T))))
    # print('V', V)
    u, s, v = LA.svd(V)
    if k < 10:
        print('Y', y)
        print('V', V)
        print('LA.pinv(V)', LA.pinv(V))
        print('V*LA.pinv(V)', np.matmul(V, LA.pinv(V)))
        print('LA.pinv(V)*V', np.matmul(LA.pinv(V), V))
        #print('Inverse', np.matmul(LA.inv(np.matmul(V.T, V)), V.T))
        # print('norm(VVt)', LA.norm(np.matmul(V, LA.pinv(V))))
    #print('Mk', Mk)
    #print('Diff', LA.norm(y - np.matmul(Mk, V)))
    # print('-------------------------------------')

    return Mk, s, LA.norm(np.eye(number_columns_V) - np.matmul(LA.pinv(V), V)), LA.norm(np.eye(number_rows_V) - np.matmul(V, LA.pinv(V))), LA.norm(y - np.matmul(Mk, V)), V

    #return np.concatenate((Mk, np.zeros([output_dimension, output_dimension])), axis = 1)