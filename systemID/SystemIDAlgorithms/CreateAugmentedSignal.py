"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""



import numpy as np

from systemID.ClassesGeneral.ClassSignal import DiscreteSignal
from systemID.SparseIDAlgorithms.GeneratePolynomialBasisFunctions import generatePolynomialBasisFunctions
from systemID.SparseIDAlgorithms.GeneratePolynomialIndex import generatePolynomialIndex

def createAugmentedSignalPolynomialBasisFunctions(original_signal, order, post_treatment, max_order):

    # Dimension
    dimension = original_signal.dimension

    # Generate Index
    index = generatePolynomialIndex(dimension, order, post_treatment, max_order=max_order)

    # Generate Polynomial Basis Functions
    lifting_functions = generatePolynomialBasisFunctions(dimension, index)

    # Construct Data for Augmented Signal
    augmented_dimension = len(lifting_functions) - 1
    data = np.zeros([augmented_dimension, original_signal.number_steps])
    data[0:dimension, :] = original_signal.data
    i = dimension
    for j in range(1, len(lifting_functions)):
        if np.sum(index[j]) > 1:
            data[i, :] = lifting_functions[j](original_signal.data)
            i += 1

    # Construct Augmented Signal
    augmented_signal = DiscreteSignal(augmented_dimension, original_signal.total_time, original_signal.frequency, signal_shape='External', data=data)

    return augmented_signal


def createAugmentedSignalWithGivenFunctions(original_signal, given_functions):

    # Dimension
    dimension = original_signal.dimension

    # Construct Data for Augmented Signal
    # augmented_dimension = dimension + len(given_functions)
    augmented_dimension = len(given_functions)
    data = np.zeros([augmented_dimension, original_signal.number_steps])
    # data[0:dimension, :] = original_signal.data
    # for i in range(dimension, augmented_dimension):
    for i in range(augmented_dimension):
        # data[i, :] = given_functions[i - dimension](original_signal.data)
        data[i, :] = given_functions[i](original_signal.data)

    # Construct Augmented Signal
    augmented_signal = DiscreteSignal(augmented_dimension, original_signal.total_time, original_signal.frequency, signal_shape='External', data=data)

    return augmented_signal
