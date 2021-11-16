"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


import numpy as np
from numpy import linalg as LA


def observerKalmanIdentificationAlgorithm(input_signal, output_signal, **kwargs):
    """
        Purpose:
            Compute the coefficients :math:`h_i`, called system Markov parameters, of the weighting sequence description
            :math:`\\boldsymbol{y}_k = CA^k\\boldsymbol{x}_0 + \displaystyle\sum_{i=0}^kh_i\\boldsymbol{u}_{k-i}`.

        Parameters:
            - **input_signal** (``DiscreteSignal``): the input signal.
            - **output_signal** (``DiscreteSignal``): the output signal.
            - **number_of_parameters** (``int``, optional): number :math:`d` of system Markov parameters to consider as non-zero in the \
            weighting sequence description. If not specified, **number_of_parameters = output_signal.number_steps**.
            - **stable_order** (``int``, optional): the order :math:`d'` such that :math:`CA^{d'}\\boldsymbol{x}_0 \\simeq 0`. If not specified, \
            **stable_order = 0**.

        Returns:
            - **markov_parameters** (``list``): list of system Markov parameters

        Imports:
            - ``import numpy as np``

        Description:
            The weighting sequence description (I/O relationship of a linear system) is

            .. math::

                \\boldsymbol{y}_k = CA^k\\boldsymbol{x}_0 + \displaystyle\sum_{i=0}^kh_i\\boldsymbol{u}_{k-i}.

            For zero initial condition, :math:`\\boldsymbol{x}_0 = 0`, the system Markov parameters :math:`h_i` appear linearly and it is possible to write in a matrix form

            .. math::

                \\boldsymbol{y} = \\boldsymbol{Y}\\boldsymbol{U} \Leftrightarrow \\boldsymbol{Y} = \\boldsymbol{y}\\boldsymbol{U}^\dagger,

            given :math:`\\boldsymbol{U}` full rank, with

            .. math::
                :nowrap:

                \\begin{align}
                    \\boldsymbol{y} & = \\begin{bmatrix} \\boldsymbol{y}_0 & \\boldsymbol{y}_1 & \\boldsymbol{y}_2 & \cdots & \\boldsymbol{y}_{l-1} \\end{bmatrix}, \\\\
                    \\boldsymbol{Y} & = \\begin{bmatrix} D & CB & CAB & \cdots & CA^{l-2}B \\end{bmatrix}, \\\\
                    \\boldsymbol{U} & = \\begin{bmatrix}
                        \\boldsymbol{u}_0 & \\boldsymbol{u}_1 & \\boldsymbol{u}_2 & \cdots & \\boldsymbol{u}_{l-1}\\\\
                                          & \\boldsymbol{u}_0 & \\boldsymbol{u}_1 & \cdots & \\boldsymbol{u}_{l-2}\\\\
                                          &                   & \\boldsymbol{u}_0 & \cdots & \\boldsymbol{u}_{l-3}\\\\
                                          &                   &                   & \ddots & \\vdots\\\\
                                          &                   &                   &        & \\boldsymbol{u}_0
                        \\end{bmatrix}.
                \\end{align}

            If **number_of_parameters** (:math:`d`) and/or **stable_order** (:math:`d'`) are specified, matrices :math:`\\boldsymbol{y}`, :math:`\\boldsymbol{Y}` \
            and :math:`\\boldsymbol{U}` become

            .. math::
                :nowrap:

                \\begin{align}
                    \\boldsymbol{y} & = \\begin{bmatrix} \\boldsymbol{y}_{d'} & \\boldsymbol{y}_{d'+1} & \\boldsymbol{y}_{d'+2} & \cdots & \\boldsymbol{y}_{l-1} \\end{bmatrix}, \\\\
                    \\boldsymbol{Y} & = \\begin{bmatrix} D & CB & CAB & \cdots & CA^{d-2}B \\end{bmatrix}, \\\\
                    \\boldsymbol{U} & = \\begin{bmatrix}
                        \\boldsymbol{u}_{d'} & \\boldsymbol{u}_{d'+1} & \\boldsymbol{u}_{d'+2} & \cdots  & \\boldsymbol{u}_{d-1}      & \cdots  & \\boldsymbol{u}_{l-1}   \\\\
                        \\vdots              & \\vdots                & \\vdots                & \\vdots & \\vdots                  & \\vdots & \\vdots                 \\\\
                        \\boldsymbol{u}_{0}  & \\boldsymbol{u}_{1}    & \\boldsymbol{u}_{2}    & \cdots  & \\boldsymbol{u}_{d-d'-1}   & \cdots  & \\boldsymbol{u}_{l-d'-1}\\\\
                                             & \\boldsymbol{u}_0      & \\boldsymbol{u}_1      & \cdots  & \\boldsymbol{u}_{d-d'-2} & \cdots  & \\boldsymbol{u}_{l-d'-2}\\\\
                                             &                        & \\boldsymbol{u}_0      & \cdots  & \\boldsymbol{u}_{d-d'-3} & \cdots  & \\boldsymbol{u}_{l-d'-3}\\\\
                                             &                        &                        & \ddots  & \\vdots                  & \\vdots & \\vdots                 \\\\
                                             &                        &                        &         & \\boldsymbol{u}_0        & \cdots  & \\boldsymbol{u}_{l-d}
                        \\end{bmatrix}.
                \\end{align}

            Notice that if :math:`d = l` and :math:`d' = 0`, this formulation is identical to the one above.

        See Also:
            - :py:mod:`~SystemIDAlgorithms.ObserverKalmanIdentificationAlgorithmWithObserver.observerKalmanIdentificationAlgorithmWithObserver`

        """

    # Get data from Signals
    y = output_signal.data
    u = input_signal.data

    # Get dimensions
    input_dimension = input_signal.dimension

    # Get number of Markov parameters to compute
    number_steps = output_signal.number_steps
    number_of_parameters = min(kwargs.get('number_of_parameters', number_steps), number_steps)
    stable_order = kwargs.get('stable_order', 0)

    # Build matrix U
    U = np.zeros([input_dimension * number_of_parameters, number_steps - stable_order])
    for i in range(0, number_of_parameters):
        U[i * input_dimension:(i + 1) * input_dimension, max(0, i - stable_order):number_steps - stable_order] = u[:, stable_order - min(i, stable_order):number_steps - i]

    # Get Y
    Y = np.matmul(y[:, stable_order:], LA.pinv(U))

    # Get Markov parameters
    markov_parameters = [Y[:, 0:input_dimension]]
    for i in range(number_of_parameters - 1):
        markov_parameters.append(Y[:, i * input_dimension + input_dimension:(i + 1) * input_dimension + input_dimension])

    return markov_parameters, U, y[:, stable_order:]
