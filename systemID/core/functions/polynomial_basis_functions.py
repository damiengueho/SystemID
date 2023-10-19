"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 25
"""

import numpy

def polynomial_basis_functions(index: numpy.ndarray,
                               inverse: bool = False):
    """
        Purpose:
            The purpose of this function is to create a list of polynomial basis functions given an **index** array.
            The **index** array provides the information about the power of each monomial for each basis function.

        Parameters:
            - **index** (``numpy.array``): the index array. **index** can be created with :py:mod:`~core.functions.polynomial_index.polynomial_index`.
            - **inverse** (``bool``, optional): if ``True``, it creates the inverse polynomial basis functions. ``False`` by default.

        Returns:
            - **basis_functions** (``list``): list of basis functions.

        Imports:
            - ``import numpy``

        Description:
            Given the **index** array, the program iterates through its rows and columns to build the corresponding basis functions. For example,
            consider an **index** array such that the input vector is of dimension 2 (number of columns of **index**) and the resulting basis functions
            are all the permutation of monomials up to degree 3:

            .. math::

                \\begin{array}{|c|c|}
                    \\hline
                    0 & 0 \\\\
                    \\hline
                    1 & 0 \\\\
                    \\hline
                    2 & 0 \\\\
                    \\hline
                    3 & 0 \\\\
                    \\hline
                    0 & 1 \\\\
                    \\hline
                    1 & 1 \\\\
                    \\hline
                    2 & 1 \\\\
                    \\hline
                    0 & 2 \\\\
                    \\hline
                    1 & 2 \\\\
                    \\hline
                    0 & 3 \\\\
                    \\hline
                \\end{array}

            The program will create a list of 10 basis functions (the number of rows of index) such that if :math:`\\boldsymbol{x} = \\begin{bmatrix} x_1 & x_2 \\end{bmatrix}^T`, then

            .. math::

                \\boldsymbol{\Phi} = \\left[1, x_1, x_1^2, x_1^3, x_2, x_1x_2, x_1^2x_2, x_2^2, x_1x_2^2, x_2^3\\right].

            Note that basis function :math:`\\phi(\\boldsymbol{x}) = 1` is included as the first basis function with degree :math:`0` in :math:`x_1` and :math:`0` in :math:`x_2`.
            For more information on how to automatically build the **index** array, see :py:mod:`~core.functions.polynomial_index.polynomial_index`.

        See Also:
            - :py:mod:`~systemID.core.functions.polynomial_index.polynomial_index`
            - :py:mod:`~systemID.core.functions.augment_data.augment_data_with_polynomial_basis_functions`
    """

    # Check arguments
    if not isinstance(index, numpy.ndarray) or index.ndim != 2:
        raise ValueError("index must be a numpy.ndarray of shape (index_length, dimension)")

    if not isinstance(inverse, bool):
        raise ValueError("inverse must be a bool")


    index_length, dimension = index.shape
    basis_functions = []

    if inverse:
        mult = -1
    else:
        mult = 1

    def make_phix(I):
        def phix(x):
            temp = 1
            for k in range(dimension):
                temp = temp*x[k]**(mult * I[k])
            return temp
        return phix

    for i in range(index_length):
        basis_functions.append(make_phix(index[i, 0:dimension]))

    return basis_functions
