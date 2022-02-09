"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""


import numpy as np
from numpy import linalg as LA


def observerKalmanIdentificationAlgorithmWithObserver(input_signal, output_signal, **kwargs):
    """
        Purpose:
            Compute the coefficients :math:`\\bar{h}_i`, called observer Markov parameters, of the weighting sequence description \
            with observer :math:`\\boldsymbol{y}_k = C\\bar{A}^k\\boldsymbol{x}_0 + \displaystyle\sum_{i=0}^k\\bar{h}_i\\boldsymbol{v}_{k-i}`.

        Parameters:
            - **input_signal** (``DiscreteSignal``): the input signal.
            - **output_signal** (``DiscreteSignal``): the output signal.
            - **observer_order** (``int``, optional): number :math:`d` of observer Markov parameters to consider as non-zero in the \
            weighting sequence description. If not specified, **observer_order = output_signal.number_steps**.
            - **stable_order** (``int``, optional): the order :math:`d'` such that :math:`C\\bar{A}^{d'}\\boldsymbol{x}_0 \\simeq 0`. If not specified, \
            **stable_order = 0**.

        Returns:
            - **observer_markov_parameters** (``list``): list of observer Markov parameters

        Imports:
            - ``import numpy as np``

        Description:
            The weighting sequence description (I/O relationship of a linear system) with observer is

            .. math::

                \\boldsymbol{y}_k = C\\bar{A}^k\\boldsymbol{x}_0 + \displaystyle\sum_{i=0}^k\\bar{h}_i\\boldsymbol{v}_{k-i}.

            For zero initial condition, :math:`\\boldsymbol{x}_0 = 0`, the observer Markov parameters :math:`\\bar{h}_i` appear linearly and it is possible to write in a matrix form

            .. math::

                \\boldsymbol{y} = \\boldsymbol{Y}\\boldsymbol{U} \Leftrightarrow \\boldsymbol{Y} = \\boldsymbol{y}\\boldsymbol{U}^\dagger,

            given :math:`\\boldsymbol{U}` full rank, with

            .. math::
                :nowrap:

                \\begin{align}
                    \\boldsymbol{y} & = \\begin{bmatrix} \\boldsymbol{y}_0 & \\boldsymbol{y}_1 & \\boldsymbol{y}_2 & \cdots & \\boldsymbol{y}_{l-1} \\end{bmatrix}, \\\\
                    \\boldsymbol{Y} & = \\begin{bmatrix} D & C\\bar{B} & C\\bar{A}\\bar{B} & \cdots & C\\bar{A}^{l-2}\\bar{B} \\end{bmatrix}, \\\\
                    \\boldsymbol{U} & = \\begin{bmatrix}
                        \\boldsymbol{u}_0 & \\boldsymbol{u}_1 & \\boldsymbol{u}_2 & \cdots & \\boldsymbol{u}_{l-1}\\\\
                                          & \\boldsymbol{v}_0 & \\boldsymbol{v}_1 & \cdots & \\boldsymbol{v}_{l-2}\\\\
                                          &                   & \\boldsymbol{v}_0 & \cdots & \\boldsymbol{v}_{l-3}\\\\
                                          &                   &                   & \ddots & \\vdots\\\\
                                          &                   &                   &        & \\boldsymbol{v}_0
                        \\end{bmatrix}.
                \\end{align}

            If **observer_order** (:math:`d`) and/or **stable_order** (:math:`d'`) are specified, matrices :math:`\\boldsymbol{y}`, :math:`\\boldsymbol{Y}` \
            and :math:`\\boldsymbol{U}` become

            .. math::
                :nowrap:

                \\begin{align}
                    \\boldsymbol{y} & = \\begin{bmatrix} \\boldsymbol{y}_{d'} & \\boldsymbol{y}_{d'+1} & \\boldsymbol{y}_{d'+2} & \cdots & \\boldsymbol{y}_{l-1} \\end{bmatrix}, \\\\
                    \\boldsymbol{Y} & = \\begin{bmatrix} D & C\\bar{B} & C\\bar{A}\\bar{B} & \cdots & C\\bar{A}^{d-1}\\bar{B} \\end{bmatrix}, \\\\
                    \\boldsymbol{U} & = \\begin{bmatrix}
                        \\boldsymbol{u}_{d'} & \\boldsymbol{u}_{d'+1} & \\boldsymbol{u}_{d'+2} & \cdots  & \\boldsymbol{u}_{d-1}      & \cdots  & \\boldsymbol{u}_{l-1}   \\\\
                        \\vdots              & \\vdots                & \\vdots                & \\vdots & \\vdots                  & \\vdots & \\vdots                 \\\\
                        \\boldsymbol{u}_{0}  & \\boldsymbol{v}_{1}    & \\boldsymbol{v}_{2}    & \cdots  & \\boldsymbol{v}_{d-d'-1}   & \cdots  & \\boldsymbol{v}_{l-d'-1}\\\\
                                             & \\boldsymbol{v}_0      & \\boldsymbol{v}_1      & \cdots  & \\boldsymbol{v}_{d-d'-2} & \cdots  & \\boldsymbol{v}_{l-d'-2}\\\\
                                             &                        & \\boldsymbol{v}_0      & \cdots  & \\boldsymbol{v}_{d-d'-3} & \cdots  & \\boldsymbol{v}_{l-d'-3}\\\\
                                             &                        &                        & \ddots  & \\vdots                  & \\vdots & \\vdots                 \\\\
                                             &                        &                        &         & \\boldsymbol{v}_0        & \cdots  & \\boldsymbol{v}_{l-d}
                        \\end{bmatrix}.
                \\end{align}

            Notice that if :math:`d = l` and :math:`d' = 0`, this formulation is identical to the one above.

        See Also:
            - :py:mod:`~SystemIDAlgorithms.ObserverKalmanIdentificationAlgorithm.observerKalmanIdentificationAlgorithm`

        """

    # Get data from Signals
    y = output_signal.data
    u = input_signal.data

    # Get dimensions
    input_dimension = input_signal.dimension
    output_dimension = output_signal.dimension

    # Get number of Markov parameters to compute
    number_steps = output_signal.number_steps
    observer_order = min(kwargs.get('observer_order', number_steps - 1), number_steps - 1)
    stable_order = kwargs.get('stable_order', 0)

    # Build matrix U
    U = np.zeros([(input_dimension + output_dimension) * observer_order + input_dimension, number_steps - stable_order])
    U[0 * input_dimension:(0 + 1) * input_dimension, :] = u[:, stable_order:number_steps]
    for i in range(1, observer_order + 1):
        U[(i - 1) * (input_dimension + output_dimension) + input_dimension:i * (input_dimension + output_dimension) + input_dimension, max(0, i - stable_order):number_steps - stable_order] = np.concatenate((u[:, stable_order - min(i, stable_order):number_steps - i], y[:, stable_order - min(i, stable_order):number_steps - i]), axis=0)

    # Get Y
    Y = np.matmul(y[:, stable_order:], LA.pinv(U))

    print(LA.norm(y[:, stable_order:] - np.matmul(Y, U)))

    # Get observer Markov parameters
    observer_markov_parameters = [Y[:, 0:input_dimension]]
    for i in range(observer_order):
        observer_markov_parameters.append(Y[:, i * (input_dimension + output_dimension) + input_dimension:(i + 1) * (input_dimension + output_dimension) + input_dimension])

    return observer_markov_parameters, y, U
