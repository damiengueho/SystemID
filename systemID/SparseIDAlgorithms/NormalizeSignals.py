"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""



import numpy as np


def normalizeSignals(signals):

    number_signals = len(signals)
    dimension = signals[0].dimension

    minb = np.zeros(dimension)
    maxb = np.zeros(dimension)

    for i in range(dimension):
        stack = signals[0].data[i, :]
        if number_signals > 1:
            for j in range(1, number_signals):
                stack = np.concatenate((stack, signals[j].data[i, :]))
        minb[i] = np.min(stack)
        maxb[i] = np.max(stack)

    for j in range(number_signals):
        for i in range(dimension):
            signals[j].data[i, :] = (signals[j].data[i, :] - (maxb[i] + minb[i]) / 2) / ((maxb[i] - minb[i]) / 2)

    return signals


