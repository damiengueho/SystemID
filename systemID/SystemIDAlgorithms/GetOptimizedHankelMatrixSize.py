"""
Author: Damien GUEHO
Copyright: Copyright (C) 2022 Damien GUEHO
License: Public Domain
Version: 23
Date: April 2022
Python: 3.7.7
"""



import numpy as np


def getOptimizedHankelMatrixSize(assumed_order, output_dimension, input_dimension):
    """
        Purpose:
            This algorithm provides the minimum number of row blocks, :math:`p`, and column blocks, :math:`q`, of the
            Hankel matrix to be full rank. This is in the context of model reduction, where the matrix is filled with Markov parameters
            of a linear system.

        Parameters:
            - **assumed_order** (``int``): the assumed rank, :math:`n`, of the Hankel matrix.
            - **output_dimension** (``int``): the output dimension, :math:`m`, of the linear system.
            - **input_dimension** (``int``): the input dimension, :math:`r`, of the linear system.

        Returns:
            - **p** (``int``): the minimum number of row blocks, :math:`p`, of the Hankel matrix.
            - **q** (``int``): the minimum number of column blocks, :math:`q`, of the Hankel matrix.

        Imports:
            - ``import numpy as np``

        Description:
            The sizes :math:`p` and :math:`q` are calculated such that :math:`pm \geq n` and :math:`qr \geq n`:

            .. math::

                p = \\left\\lceil \\dfrac{n}{m} \\right\\rceil, \quad \quad \quad q = \\left\\lceil\\dfrac{n}{r}\\right\\rceil.

        See Also:
            -
    """

    p = int(np.ceil(assumed_order / output_dimension))
    q = int(np.ceil(assumed_order / input_dimension))

    return p, q