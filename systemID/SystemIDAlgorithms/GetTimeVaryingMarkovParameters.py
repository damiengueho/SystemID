"""
Author: Damien GUEHO
Copyright: Copyright (C) 2022 Damien GUEHO
License: Public Domain
Version: 23
Date: April 2022
Python: 3.7.7
"""


import numpy as np


def getTimeVaryingMarkovParameters(A, B, C, D, tk, dt, **kwargs):
    """
        Purpose:
            This program computes time-varying Markov parameters at time **tk** for a certain number of steps **number_steps**, if specified.

        Parameters:
            - **A** (``fun``): the system matrix.
            - **B** (``fun``): the input influence matrix.
            - **C** (``fun``): the output influence matrix.
            - **D** (``fun``): the direct transmission matrix.
            - **tk** (``float``): the time step at which the time-varying Markov parameters are computed.
            - **dt** (``float``): the length of the time step for which the zero-order hold approximation for the system matrices is valid.
            - **number_steps** (``int``, *optional*): the number :math:`p` of time-varying Markov parameters to compute. If not specified, it will calculate the maximum amount of time-varying Markov parameters possible.

        Returns:
            - **time_varying_markov_parameters** (``list``): a list of the calculated time-varying Markov parameters.

        Imports:
            - ``import numpy as np``

        Description:
            The list of the time-varying Markov parameters at time **tk** is

            .. math::

                \\left[D_k, C_kB_{k-1}, C_kA_{k-1}B_{k-2}, C_kA_{k-1}A_{k-2}B_{k-3}, \\cdots, C_kA_{k-1}\\cdots A_1B_0\\right].

            If the number :math:`p` of time-varying Markov parameters to compute, **number_steps**, is specified, the list is
            truncated to reflect only the first :math:`p` time-varying Markov parameters.

        See Also:
            - :py:mod:`~SystemIDAlgorithms.GetTimeVaryingMarkovParameters.getTimeVaryingMarkovParameters_matrix`
    """

    number_steps = kwargs.get('number_steps', int(np.round(tk / dt + 1)))

    if number_steps <= 0:
        return []
    elif number_steps == 1:
        return [D(tk)]
    else:
        time_varying_markov_parameters = [D(tk)]
        temp = C(tk)
        for i in range(1, number_steps):
            time_varying_markov_parameters.append(np.matmul(temp, B(tk - i * dt)))
            temp = np.matmul(temp, A(tk - i * dt))
        return time_varying_markov_parameters


def getTimeVaryingMarkovParameters_matrix(A, B, C, D, k, **kwargs):
    """
        Purpose:
            This program computes time-varying Markov parameters at time **tk** for a certain number of steps **number_steps**, if specified.
            The output of this program is identical to :py:mod:`~SystemIDAlgorithms.GetTimeVaryingMarkovParameters.getTimeVaryingMarkovParameters`
            with the only difference that the input matrices are given as ``ndarray`` instead of ``fun``.

        Parameters:
            - **A** (``ndarray``): the system matrix.
            - **B** (``ndarray``): the input influence matrix.
            - **C** (``ndarray``): the output influence matrix.
            - **D** (``ndarray``): the direct transmission matrix.
            - **k** (``float``): the time step at which the time-varying Markov parameters are computed.
            - **number_steps** (``int``, *optional*): the number :math:`p` of time-varying Markov parameters to compute. If not specified, it will calculate the maximum amount of time-varying Markov parameters possible.

        Returns:
            - **time_varying_markov_parameters** (``list``): a list of the calculated time-varying Markov parameters.

        Imports:
            - ``import numpy as np``

        Description:
            The list of the time-varying Markov parameters at time **k** is

            .. math::

                \\left[D_k, C_kB_{k-1}, C_kA_{k-1}B_{k-2}, C_kA_{k-1}A_{k-2}B_{k-3}, \\cdots, C_kA_{k-1}\\cdots A_1B_0\\right].

            If the number :math:`p` of time-varying Markov parameters to compute, **number_steps**, is specified, the list is
            truncated to reflect only the first :math:`p` time-varying Markov parameters.

        See Also:
            - :py:mod:`~SystemIDAlgorithms.GetTimeVaryingMarkovParameters.getTimeVaryingMarkovParameters`
    """

    number_steps = kwargs.get('number_steps', int(np.round(k + 1)))

    if number_steps <= 0:
        return []
    elif number_steps == 1:
        return [D[:, :, k]]
    else:
        time_varying_markov_parameters = [D[:, :, k]]
        temp = C[:, :, k]
        for i in range(1, number_steps):
            time_varying_markov_parameters.append(np.matmul(temp, B[:, :, k - i]))
            temp = np.matmul(temp, A[:, :, k - i])
        return time_varying_markov_parameters
