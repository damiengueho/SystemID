"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""



import numpy as np


def scale_each_dimension(signals):

    number_signals = len(signals)
    dimension = signals[0].dimension

    min_max = np.zeros([2, dimension])

    for i in range(dimension):
        stack = signals[0].data[i, :]
        for j in range(1, number_signals):
            stack = np.concatenate((stack, signals[j].data[i, :]))
        min_max[0, i] = np.min(stack)
        min_max[1, i] = np.max(stack)

    for j in range(number_signals):
        for i in range(dimension):
            signals[j].data[i, :] = (signals[j].data[i, :] - (min_max[1, i] + min_max[0, i]) / 2) / ((min_max[1, i] - min_max[0, i]) / 2)

    return signals, min_max


# def scale_global(signals):
#
#     number_signals = len(signals)
#     dimension = signals[0].dimension
#
#     min_max = np.zeros([2,])
#
#     for i in range(dimension):
#         stack = signals[0].data[i, :]
#         for j in range(1, number_signals):
#             stack = np.concatenate((stack, signals[j].data[i, :]))
#         min_max[0, i] = np.min(stack)
#         min_max[1, i] = np.max(stack)
#
#     for j in range(number_signals):
#         for i in range(dimension):
#             signals[j].data[i, :] = (signals[j].data[i, :] - (min_max[1, i] + min_max[0, i]) / 2) / ((min_max[1, i] - min_max[0, i]) / 2)
#
#     return signals, min_max


def unscale(signals, min_max):

    number_signals = len(signals)
    dimension = signals[0].dimension

    for j in range(number_signals):
        for i in range(dimension):
            signals[j].data[i, :] = signals[j].data[i, :] * (min_max[1, i] - min_max[0, i]) / 2 + (min_max[1, i] + min_max[0, i]) / 2

    return signals
