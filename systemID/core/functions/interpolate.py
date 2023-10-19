"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 25
"""


import numpy
import scipy

def interpolate(tspan: numpy.ndarray,
                tspans: list,
                data: list,
                b_spline_degree: int = 3):
    """
        Purpose:
            Interpolate the data using B splines..

        Parameters:
            - **tspan** (``numpy.ndarray``): a 1-dimensional ``numpy.ndarray`` of shape
            (number_steps, ) that represents the target time span.
            - **tspans** (``list``): a list that contains the original time spans. Each time span is
            a ``numpy.ndarray``.
            - **data** (``list``): a list containing the data to be interpolated. Each data is
            a ``numpy.ndarray`` of shape (dimension, number_steps), with number_steps matching the corresponding
            time span in **tspans**.
            - **b_spline_degree** (``int``, optional): the spline degree. If not specified, default is 3.

        Returns:
            - **interpolated_data** (``numpy.ndarray``): a 3-dimensional ``numpy.ndarray`` of shape
            (dimension, number_steps, number_experiments) containing the data to be interpolated.
            - **interpolant_functions** (``list``): the interpolant functions.

        Imports:
            - ``import numpy``
            - ``import scipy``

        Description:
            This program ...

        See Also:
            -
    """

    number_signals = len(data)
    interpolated_data = []
    interpolant_functions = []

    for i in range(number_signals):
        interpolant_function = scipy.interpolate.make_interp_spline(tspans[i], data[i], k=b_spline_degree, axis=1)
        interpolant_functions.append(interpolant_function)
        interpolated_data.append(interpolant_function(tspan))

    return interpolated_data, interpolant_functions
