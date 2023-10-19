"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 25
"""


import numpy

def polynomial_index(dimension: int,
                     order: int,
                     max_order: int = -1):
    """
        Purpose:
            This function generates all permutations of monomials up to a certain order. It concatenates all the permutations in
            an array, where the number of rows is the number of resulting permutations and the number of columns is the number
            of monomials (or the **dimension** of the input vector).

        Parameters:
            - **dimension** (``int``): number of monomials.
            - **order** (``int``): maximum order for a given monomial.
            - **max_order** (``int``, optional): maximum order for a given basis function.

        Returns:
            - **index** (``numpy.array``): the index array.

        Imports:
            - ``import numpy``

        Description:
            Iterating through the **dimension** and the **order**, the program builds an array that contains all the permutations of
            1-D monomials. For example, if **dimension** :math:`=2` and **order** :math:`=3`, the program creates the array :math:`I_1`,
            resulting in 16 permutations. If **max_order** :math:`=3` is specified (let's say we want the possible order of monomials
            to be up to :math:`3`, but the resulting basis functions to be of degree at most :math:`3`), the program deletes the rows
            with a resulting degree greater than :math:`3`. This would result in the array :math:`I_2`.

            .. math::

                I_1 = \\begin{array}{|c|c|}
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
                    3 & 1 \\\\
                    \\hline
                    0 & 2 \\\\
                    \\hline
                    1 & 2 \\\\
                    \\hline
                    2 & 2 \\\\
                    \\hline
                    3 & 2 \\\\
                    \\hline
                    0 & 3 \\\\
                    \\hline
                    1 & 3 \\\\
                    \\hline
                    2 & 3 \\\\
                    \\hline
                    3 & 3 \\\\
                    \\hline
                \\end{array}, \quad \quad I_2 = \\begin{array}{|c|c|}
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
                \\end{array}.


            This resulting **index** array can be use as an argument of :py:mod:`~core.functions.polynomial_basis_functions.polynomial_basis_functions`
            to create the resulting polynomial basis functions.


        See Also:
            - :py:mod:`~systemID.core.functions.polynomial_basis_functions.polynomial_basis_functions`
            - :py:mod:`~systemID.core.functions.augment_data.augment_data_with_polynomial_basis_functions`
        """

    # Check arguments
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("dimension must be a positive integer")

    if not isinstance(order, int) or order <= 0:
        raise ValueError("order must be a positive integer")

    if not isinstance(max_order, int):
        raise ValueError("max_order must be a positive or zero integer; if negative it will not be taken into account")

    a = numpy.ones([dimension])
    order = order + a

    index = numpy.zeros([int(order[0]), 1])
    for i in range(int(order[0])):
        index[i, 0] = i + 1

    for i in range(1, dimension):
        repel = index
        repsize = len(index[:, 0])
        repwith = numpy.ones([repsize, 1])

        for j in range(2, int(order[i]) + 1):
            repwith = numpy.concatenate((repwith, numpy.ones([repsize, 1]) * j), axis=0)

        index = numpy.concatenate((numpy.kron(numpy.ones((int(order[i]), 1)), repel), repwith), axis=1)

    s = index.shape
    o = numpy.ones([s[0], s[1]])
    index = index - o
    index = index.astype(int)

    if max_order >= 0:
        rows = []
        for i in range(s[0]):
            if numpy.sum(index[i, :]) > max_order:
                rows.append(i)

        index = numpy.delete(index, rows, axis=0)

    return index
