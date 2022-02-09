"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""



import numpy as np

from systemID.ClassesGeneral.ClassSignal import DiscreteSignal
from systemID.SparseIDAlgorithms.GeneratePolynomialBasisFunctions import generatePolynomialBasisFunctions
from systemID.SparseIDAlgorithms.GeneratePolynomialIndex import generatePolynomialIndex

def createAugmentedSignalPolynomialBasisFunctions(original_signal, order, post_treatment, max_order):
    """
        Purpose:
            Create an augmented signal appending polynomial functions of original data.

        Parameters:
            - **original_signal** (``DiscreteSignal``): the signal to be augmented.
            - **order** (``int``): the order of single monomials to be appended.
            - **post_treatment** (``bool``): boolean to know whether or not **max_order** is applied.
            - **max_order** (``int``): the maximum order of polynomials to be appended.

        Returns:
            - **augmented_signal** (``DiscreteSignal``): the augmented signal.

        Imports:
            - ``import numpy as np``
            - ``from systemID.ClassesGeneral.ClassSignal import DiscreteSignal``
            - ``from systemID.SparseIDAlgorithms.GeneratePolynomialBasisFunctions import generatePolynomialBasisFunctions``
            - ``from systemID.SparseIDAlgorithms.GeneratePolynomialIndex import generatePolynomialIndex``

        Description:
            This program first generates the index of orders useful for creating the polynomial basis functions. If \
            **post_treatment == True**, the total order for any polynomial basis function is **max_order**. For example, \
            in dimension 2, polynomials function of :math:`x_1` and :math:`x_2` when **max_order = 3** will be

            .. math::

                1 \quad x_1 \quad x_2 \quad x_1^2 \quad x_1x_2 \quad x_2^2 \quad x_1^3 \quad x_1^2x_2 \quad x_1x_2^2 \quad x_2^3.

            Note that :math:`x_1^3x_2` for example will not be included because the order of this polynomial is 4 and is \
            beyond **max_order**.\\

            Basis function :math:`1` is never included.

        See Also:
            - :py:mod:`~ClassesGeneral.ClassSignal.DiscreteSignal`
            - :py:mod:`~SparseIDAlgorithms.GeneratePolynomialBasisFunctions.generatePolynomialBasisFunctions`
            - :py:mod:`~SparseIDAlgorithms.GeneratePolynomialIndex.generatePolynomialIndex`
            - :py:mod:`~SystemIDAlgorithms.CreateAugmentedSignal.createAugmentedSignalWithGivenFunctions`
        """

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
    """
        Purpose:
            Create an augmented signal appending given functions of original data.

        Parameters:
            - **original_signal** (``DiscreteSignal``): the signal to be augmented.
            - **given_functions** (``list``): a list of functions.

        Returns:
            - **augmented_signal** (``DiscreteSignal``): the augmented signal.

        Imports:
            - ``import numpy as np``
            - ``from systemID.ClassesGeneral.ClassSignal import DiscreteSignal``

        Description:
            This program create a signal with data function of the original data.

        See Also:
            - :py:mod:`~ClassesGeneral.ClassSignal.DiscreteSignal`
            - :py:mod:`~SystemIDAlgorithms.CreateAugmentedSignal.createAugmentedSignalPolynomialBasisFunctions`
    """

    # Dimension
    dimension = original_signal.dimension

    # Construct Data for Augmented Signal
    augmented_dimension = len(given_functions)
    data = np.zeros([augmented_dimension, original_signal.number_steps])
    for i in range(augmented_dimension):
        data[i, :] = given_functions[i](original_signal.data)

    # Construct Augmented Signal
    augmented_signal = DiscreteSignal(augmented_dimension, original_signal.total_time, original_signal.frequency, signal_shape='External', data=data)

    return augmented_signal
