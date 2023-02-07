"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""



import numpy as np

from systemID.signals.discrete import discrete_signal
from systemID.functions.polynomial_basis_functions import polynomial_basis_functions
from systemID.functions.polynomial_index import polynomial_index

def augment_signals_with_polynomial_basis_functions(signals, order, **kwargs):
    """
        Purpose:
            Create augmented signals appending polynomial functions of original data.

        Parameters:
            - **signals** (``list``): a list of ``discrete_signals`` to be augmented.
            - **order** (``int``): the order of single monomials to be appended.
            - **max_order** (``int``, optional): the maximum order of polynomials to be appended.

        Returns:
            - **augmented_signal** (``discrete_signal``): the augmented signals.

        Imports:
            - ``import numpy as np``
            - ``from systemID.signals.discrete import discrete_signal``
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
    dimension = signals[0].dimension
    number_signals = len(signals)

    # Generate Index
    max_order = kwargs.get('max_order', -1)
    if max_order >= 0:
        index = polynomial_index(dimension, order, max_order=max_order)
    else:
        index = polynomial_index(dimension, order)

    # Generate Polynomial Basis Functions
    lifting_functions = polynomial_basis_functions(index)
    augmented_dimension = len(lifting_functions) - 1

    augmented_signals = []

    for k in range(number_signals):

        # Construct Data for Augmented Signal
        data = np.zeros([augmented_dimension, signals[k].number_steps])
        data[0:dimension, :] = signals[k].data
        i = dimension
        for j in range(1, len(lifting_functions)):
            if np.sum(index[j]) > 1:
                data[i, :] = lifting_functions[j](signals[k].data)
                i += 1

        # Construct Augmented Signal
        augmented_signals.append(discrete_signal(data=data, frequency=signals[k].frequency))

    return augmented_signals





def augment_signals_with_given_functions(signals, given_functions):
    """
        Purpose:
            Create an augmented signals appending given functions of original data.

        Parameters:
            - **original_signal** (``DiscreteSignal``): the signals to be augmented.
            - **given_functions** (``list``): a list of functions.

        Returns:
            - **augmented_signal** (``DiscreteSignal``): the augmented signals.

        Imports:
            - ``import numpy as np``
            - ``from systemID.ClassesGeneral.ClassSignal import DiscreteSignal``

        Description:
            This program create a signals with data function of the original data.

        See Also:
            - :py:mod:`~ClassesGeneral.ClassSignal.DiscreteSignal`
            - :py:mod:`~SystemIDAlgorithms.CreateAugmentedSignal.createAugmentedSignalPolynomialBasisFunctions`
    """

    number_signals = len(signals)

    augmented_dimension = len(given_functions)
    augmented_signals = []

    for k in range(number_signals):

        # Construct Data for Augmented Signal
        data = np.zeros([augmented_dimension, signals[k].number_steps])
        for i in range(augmented_dimension):
            data[i, :] = given_functions[i](signals[k].data)

        # Construct Augmented Signal
        augmented_signals.append(discrete_signal(data=data, frequency=signals[k].frequency))

    return augmented_signals
