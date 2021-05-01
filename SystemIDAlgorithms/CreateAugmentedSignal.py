"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 10
Date: April 2021
Python: 3.7.7
"""



import numpy as np

from ClassesGeneral.ClassSignal import DiscreteSignal
from SparseIDAlgorithms.GeneratePolynomialBasisFunctions import generatePolynomialBasisFunctions
from SparseIDAlgorithms.GeneratePolynomialIndex import generatePolynomialIndex

def createAugmentedSignalPolynomialBasisFunctions(original_signal, order, post_treatment, max_order):

    # Dimension
    dimension = original_signal.dimension

    # Generate Index
    index = generatePolynomialIndex(dimension, order, post_treatment, max_order=max_order)

    # Generate Polynomial Basis Functions
    lifting_functions = generatePolynomialBasisFunctions(dimension, index)

    # Construct Data for Augmented Signal
    augmented_dimension = dimension + len(lifting_functions)
    data = np.zeros([augmented_dimension, original_signal.number_steps])
    data[0:dimension, :] = original_signal.data
    for i in range(dimension, augmented_dimension):
        data[i, :] = lifting_functions[i - dimension](original_signal.data)

    # Construct Augmented Signal
    augmented_signal = DiscreteSignal(augmented_dimension, original_signal.name + ' Augmented', original_signal.total_time, original_signal.frequency, signal_shape='External', data=data)

    return augmented_signal


def createAugmentedSignalWithGivenFunctions(original_signal, given_functions):

    # Dimension
    dimension = original_signal.dimension

    # Construct Data for Augmented Signal
    augmented_dimension = dimension + len(given_functions)
    data = np.zeros([augmented_dimension, original_signal.number_steps])
    data[0:dimension, :] = original_signal.data
    for i in range(dimension, augmented_dimension):
        data[i, :] = given_functions[i - dimension](original_signal.data)

    # Construct Augmented Signal
    augmented_signal = DiscreteSignal(augmented_dimension, original_signal.name + ' Augmented', original_signal.total_time, original_signal.frequency, signal_shape='External', data=data)

    return augmented_signal