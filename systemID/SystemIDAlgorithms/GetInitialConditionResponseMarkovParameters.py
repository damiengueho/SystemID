"""
Author: Damien GUEHO
Copyright: Copyright (C) 2022 Damien GUEHO
License: Public Domain
Version: 23
Date: April 2022
Python: 3.7.7
"""


import numpy as np


def getInitialConditionResponseMarkovParameters(A, C, number_steps):
    """
        Purpose:
            This program computes a list comprised of the first :math:`p` time-invariant initial condition Markov parameters.

        Parameters:
            - **A** (``fun``): the system matrix.
            - **C** (``fun``): the output influence matrix.
            - **number_steps** (``int``): the number :math:`p` of time-invariant initial condition Markov parameters to compute.

        Returns:
            - **markov_parameters** (``list``): a list of the first :math:`p` time-invariant initial condition Markov parameters.

        Imports:
            - ``import numpy as np``

        Description:
            The list of the first :math:`p` time-invariant initial condition Markov parameters is

            .. math::

                \\left[C, CA, CA^2, CA^3, \\cdots, CA^{p-1}\\right].


        See Also:
            - :py:mod:`~SystemIDAlgorithms.GetMarkovParameters.getMarkovParameters`
    """

    C_mat = C(0)
    A_mat = A(0)

    markov_parameters = [C_mat]
    temp = C_mat

    for i in range(1, number_steps):
        temp = np.matmul(temp, A_mat)
        markov_parameters.append(temp)

    return markov_parameters
