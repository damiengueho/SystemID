"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""


from systemID.signals.discrete import discrete_signal
import scipy.interpolate as interp


def interpolate(tspan, tspans, signals, b_spline_degree=3):

    number_signals = len(signals)
    interpolated_signals = []
    interpolant_functions = []

    for i in range(number_signals):
        interpolant_function = interp.make_interp_spline(tspans[i], signals[i].data, k=b_spline_degree, axis=1)
        interpolant_functions.append(interpolant_function)
        interpolated_signals.append(discrete_signal(frequency=signals[i].frequency, data=interpolant_function(tspan)))

    return interpolated_signals, interpolant_functions



def interpolant_functions(tspans, signals, b_spline_degree=3):

    number_signals = len(signals)
    interpolant_functions = []

    for i in range(number_signals):
        interpolant_functions.append(interp.make_interp_spline(tspans[i], signals[i].data, k=b_spline_degree, axis=1))

    return interpolant_functions
