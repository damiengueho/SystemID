"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""


from systemID.signals.discrete import discrete_signal
import scipy.interpolate as interp


def interpolate(time, times, signals, b_spline_degree=3):

    number_signals = len(signals)
    interpolated_signals = []

    for i in range(number_signals):
        interp_signal = interp.make_interp_spline(times[i], signals[i].data, k=b_spline_degree, axis=1)
        interpolated_signals.append(discrete_signal(frequency=signals[i].frequency, data=interp_signal(time)))

    return interpolated_signals