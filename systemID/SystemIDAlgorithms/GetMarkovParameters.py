"""
Author: Damien GUEHO
Copyright: Copyright (C) 2022 Damien GUEHO
License: Public Domain
Version: 23
Date: April 2022
Python: 3.7.7
"""


import numpy as np


def getMarkovParameters(A, B, C, D, number_steps):
    """
        Purpose:
            This program computes a list comprised of the first :math:`p` time-invariant Markov parameters.

        Parameters:
            - **A** (``fun``): the system matrix.
            - **B** (``fun``): the input influence matrix.
            - **C** (``fun``): the output influence matrix.
            - **D** (``fun``): the direct transmission matrix.
            - **number_steps** (``int``): the number :math:`p` of time-invariant Markov parameters to compute.

        Returns:
            - **markov_parameters** (``list``): a list of the first :math:`p` time-invariant Markov parameters.

        Imports:
            - ``import numpy as np``

        Description:
            The list of the first :math:`p` time-invariant initial condition Markov parameters is

            .. math::

                \\left[D, CB, CAB, CA^2B, \\cdots, CA^{p-2}B\\right].


        See Also:
            - :py:mod:`~SystemIDAlgorithms.GetInitialConditionResponseMarkovParameters.getInitialConditionResponseMarkovParameters`
    """

    A_mat = A(0)
    B_mat = B(0)
    C_mat = C(0)
    D_mat = D(0)

    markov_parameters = [D_mat]
    temp = C_mat

    for i in range(1, number_steps):
        markov_parameters.append(np.matmul(temp, B_mat))
        temp = np.matmul(temp, A_mat)

    return markov_parameters
