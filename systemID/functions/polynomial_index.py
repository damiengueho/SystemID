"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""


import numpy as np


def polynomial_index(dimension, order, **kwargs):
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
            - **index** (``np.array``): the index array.

        Imports:
            - ``import numpy as np``

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


            This resulting **index** array can be use as an argument of :py:mod:`~SparseIDAlgorithms.GeneratePolynomialBasisFunctions.generatePolynomialBasisFunctions`
            to create the resulting polynomial basis functions.


        See Also:
            - :py:mod:`~SparseIDAlgorithms.GeneratePolynomialBasisFunctions.generatePolynomialBasisFunctions`
        """


    a = np.ones([dimension])
    order = order + a

    index = np.zeros([int(order[0]), 1])
    for i in range(int(order[0])):
        index[i, 0] = i + 1

    for i in range(1, dimension):
        repel = index
        repsize = len(index[:, 0])
        repwith = np.ones([repsize, 1])

        for j in range(2, int(order[i]) + 1):
            repwith = np.concatenate((repwith, np.ones([repsize, 1]) * j), axis=0)

        index = np.concatenate((np.kron(np.ones((int(order[i]), 1)), repel), repwith), axis=1)

    s = index.shape
    o = np.ones([s[0], s[1]])
    index = index - o
    index = index.astype(int)

    max_order = kwargs.get('max_order', -1)

    if max_order >= 0:
        rows = []
        for i in range(s[0]):
            if np.sum(index[i, :]) > max_order:
                rows.append(i)

        index = np.delete(index, rows, axis=0)

    return index
