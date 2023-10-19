"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 25
"""


import numpy
import scipy


def observer_controller_identification_algorithm(input_data: numpy.ndarray,
                                                 feedback_data: numpy.ndarray,
                                                 output_data: numpy.ndarray,
                                                 observer_order: int = None,
                                                 stable_order: int = None
):
    """
        Purpose:
            Compute the coefficients :math:`\\tilde{h}_i`, called observer/controller Markov parameters, of the weighting sequence description \
            with observer :math:`\\boldsymbol{v}_k^f = \\bar{C}\\bar{A}^k\\boldsymbol{x}_0 + \displaystyle\sum_{i=0}^k\\tilde{h}_i\\boldsymbol{v}_{k-i}`.

        Parameters:
            - **input_signal** (``DiscreteSignal``): the input signals.
            - **feedback_signal** (``DiscreteSignal``): the feedback signals.
            - **output_signal** (``DiscreteSignal``): the output signals.
            - **observer_order** (``int``, optional): number :math:`d` of observer/controller Markov parameters to consider as non-zero in the \
            weighting sequence description. If not specified, **observer_order = output_signal.number_steps**.
            - **stable_order** (``int``, optional): the order :math:`d'` such that :math:`\\bar{C}\\bar{A}^{d'}\\boldsymbol{x}_0 \\simeq 0`. If not specified, \
            **stable_order = 0**.

        Returns:
            - **observer_controller_markov_parameters** (``list``): list of observer/controller Markov parameters

        Imports:
            - ``import numpy as np``

        Description:
            The weighting sequence description (I/O relationship of a linear system) with observer and full state feedback controller is

            .. math::

                \\boldsymbol{v}_k^f = \\bar{C}\\bar{A}^k\\boldsymbol{x}_0 + \displaystyle\sum_{i=0}^k\\tilde{h}_i\\boldsymbol{v}_{k-i}.

            For zero initial condition, :math:`\\boldsymbol{x}_0 = 0`, the observer/controller Markov parameters :math:`\tilde{h}_i` appear linearly and it is possible to write in a matrix form

            .. math::

                \\boldsymbol{v}^f = \\boldsymbol{Y}\\boldsymbol{U} \Leftrightarrow \\boldsymbol{Y} = \\boldsymbol{v}\\boldsymbol{U}^\dagger,

            given :math:`\\boldsymbol{U}` full rank, with

            .. math::
                :nowrap:

                \\begin{align}
                    \\boldsymbol{v}^f & = \\begin{bmatrix} \\boldsymbol{y}_0 & \\boldsymbol{y}_1 & \\boldsymbol{y}_2 & \cdots & \\boldsymbol{y}_{l-1} \\\\
                                                           \\boldsymbol{u}_0^f & \\boldsymbol{u}_1^f & \\boldsymbol{u}_2^f & \cdots & \\boldsymbol{u}_{l-1}^f
                                          \\end{bmatrix}, \\\\
                    \\boldsymbol{Y} & = \\begin{bmatrix} \\bar{D} & \\bar{C}\\bar{B} & \\bar{C}\\bar{A}\\bar{B} & \cdots & \\bar{C}\\bar{A}^{l-2}\\bar{B} \\end{bmatrix}, \\\\
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
                    \\boldsymbol{y} & = \\begin{bmatrix} \\boldsymbol{y}_{d'} & \\boldsymbol{y}_{d'+1} & \\boldsymbol{y}_{d'+2} & \cdots & \\boldsymbol{y}_{l-1} \\\\
                                                         \\boldsymbol{u}_{d'}^f & \\boldsymbol{u}_{d' + 1}^f & \\boldsymbol{u}_{d' + 2}^f & \cdots & \\boldsymbol{u}_{l-1}^f
                                        \\end{bmatrix}, \\\\
                    \\boldsymbol{Y} & = \\begin{bmatrix} \\bar{D} & C\\bar{B} & \\bar{C}\\bar{A}\\bar{B} & \cdots & \\bar{C}\\bar{A}^{d-1}\\bar{B} \\end{bmatrix}, \\\\
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
            - :py:mod:`~SystemIDAlgorithms.ObserverKalmanIdentificationAlgorithmWithObserver.observerKalmanIdentificationAlgorithmWithObserver`

        """

    # Return dictionary with results
    results = {}

    # Get data from Signals
    y = output_data
    u = input_data
    uf = feedback_data

    # Get dimensions
    (input_dimension, number_steps) = input_data.shape
    output_dimension = output_data.shape[0]

    # Get number of Markov parameters to compute
    observer_order = min(observer_order, number_steps - 1)
    # stable_order = kwargs.get('stable_order', 0)

    # Build matrix U
    U = numpy.zeros([(input_dimension + output_dimension) * observer_order + input_dimension, number_steps - stable_order])
    U[0 * input_dimension:(0 + 1) * input_dimension, :] = u[:, stable_order:number_steps]
    for i in range(1, observer_order + 1):
        U[(i - 1) * (input_dimension + output_dimension) + input_dimension:i * (input_dimension + output_dimension) + input_dimension, max(0, i - stable_order):number_steps - stable_order] = numpy.concatenate((u[:, stable_order - min(i, stable_order):number_steps - i], y[:, stable_order - min(i, stable_order):number_steps - i]), axis=0)

    # Get Y
    yt = numpy.concatenate((y[:, stable_order:], uf[:, stable_order:]), axis=0)
    Y = numpy.matmul(yt, scipy.linalg.pinv(U))

    # Get observer Markov parameters
    observer_controller_markov_parameters = [Y[:, 0:input_dimension]]
    for i in range(observer_order):
        observer_controller_markov_parameters.append(Y[:, i * (input_dimension + output_dimension) + input_dimension:(i + 1) * (input_dimension + output_dimension) + input_dimension])

    results['observer_controller_markov_parameters'] = observer_controller_markov_parameters
    results['U'] = U
    results['yt'] = yt
    results['Y'] = Y
    results['Error OKID'] = scipy.linalg.norm(yt - numpy.matmul(Y, U))

    return results
