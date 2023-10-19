"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 25
"""


import numpy

from systemID.core.functions.polynomial_index import polynomial_index
from systemID.core.functions.polynomial_basis_functions import polynomial_basis_functions

def augment_data_with_polynomial_basis_functions(data: numpy.ndarray,
                                                 order: int,
                                                 max_order: int = -1):
    """
        Purpose:
            Create augmented data appending polynomial functions of original data.

        Parameters:
            - **data** (``numpy.ndarray``): a 3-dimensional ``numpy.ndarray`` of shape
            (dimension, number_steps, number_experiments) containing the data to be augmented.
            - **order** (``int``): the order of single monomials to be appended.
            - **max_order** (``int``, optional): the maximum order of polynomials to be appended.

        Returns:
            - **augmented_data** (``numpy.ndarray``): a 3-dimensional ``numpy.ndarray`` of shape
            (augmented_dimension, number_steps, number_experiments) containing the augmented data.

        Imports:
            - ``import numpy``
            - ``from systemID.core.functions.polynomial_index import polynomial_index``
            - ``from systemID.core.functions.polynomial_basis_functions import polynomial_basis_functions``

        Description:
            This program first generates the index of orders useful for creating the polynomial basis functions. If \
            **max_order** is specified, the total order for any polynomial basis function is **max_order**. For example, \
            in dimension 2, polynomials function of :math:`x_1` and :math:`x_2` when **max_order = 3** will be

            .. math::

                1 \quad x_1 \quad x_2 \quad x_1^2 \quad x_1x_2 \quad x_2^2 \quad x_1^3 \quad x_1^2x_2 \quad x_1x_2^2 \quad x_2^3.

            Note that :math:`x_1^3x_2` for example will not be included because the order of this polynomial is 4 and is \
            beyond **max_order**.\\

            Basis function :math:`1` is never included.

        See Also:
            - :py:mod:`~systemID.core.functions.polynomial_index.polynomial_index`
            - :py:mod:`~systemID.core.functions.polynomial_basis_functions.polynomial_basis_functions`
            - :py:mod:`~systemID.core.functions.augment_data.augment_data_with_given_functions`
    """

    # Check arguments
    if not isinstance(data, numpy.ndarray):
        raise ValueError("data must be a numpy.ndarrays of shape (dimension, number_steps, number_experiments)")

    if not isinstance(order, int) or order <= 0:
        raise ValueError("order must be a positive integer")

    if not isinstance(max_order, int):
        raise ValueError("max_order must be a positive or zero integer; if negative it will not be taken into account")


    # Dimension
    dimension, number_steps, number_experiments = data.shape

    # Generate Index
    if max_order >= 0:
        index = polynomial_index(dimension, order, max_order=max_order)
    else:
        index = polynomial_index(dimension, order)

    # Generate Polynomial Basis Functions
    lifting_functions = polynomial_basis_functions(index)
    augmented_dimension = len(lifting_functions) - 1

    # Construct Data for Augmented Signal
    augmented_data = numpy.zeros([augmented_dimension, number_steps, number_experiments])

    for k in range(number_experiments):

        augmented_data[0:dimension, :, k] = data[:, :, k]
        i = dimension
        for j in range(1, len(lifting_functions)):
            if numpy.sum(index[j]) > 1:
                augmented_data[i, :, k] = lifting_functions[j](data[:, :, k])
                i += 1

    return augmented_data



def augment_data_with_given_functions(data: numpy.ndarray,
                                      given_functions: list):
    """
        Purpose:
            Create augmented data given functions of original data.

        Parameters:
            - **data** (``numpy.ndarray``): a 3-dimensional ``numpy.ndarray`` of shape
            (dimension, number_steps, number_experiments) to be augmented.
            - **given_functions** (``list``): a list of callable functions.

        Returns:
            - **augmented_data** (``numpy.ndarray``): a 3-dimensional ``numpy.ndarray`` of shape
            (augmented_dimension, number_steps, number_experiments) containing the augmented data.

        Imports:
            - ``import numpy``

        Description:
            This program ...

        See Also:
            - :py:mod:`~systemID.core.functions.polynomial_index import polynomial_index`
            - :py:mod:`~systemID.core.functions.polynomial_basis_functions import polynomial_basis_functions`
            - :py:mod:`~systemID.core.functions.augment_data import augment_data_with_polynomial_basis_functions`
    """

    # Check arguments
    if not isinstance(data, list):
        raise ValueError("data must be a numpy.ndarray of shape (dimension, number_steps, number_experiments)")

    if not isinstance(given_functions, list):
        raise ValueError("given_functions must be a list of callable functions")


    # Dimension
    dimension, number_steps, number_experiments = data.shape

    augmented_dimension = len(given_functions)

    # Construct Data for Augmented Signal
    augmented_data = numpy.zeros([augmented_dimension, number_steps, number_experiments])

    for k in range(number_experiments):

        for i in range(augmented_dimension):
            augmented_data[i, :, k] = given_functions[i](data[:, :, k])

    return augmented_data
