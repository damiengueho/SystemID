"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


import matplotlib.pyplot as plt
import numpy as np


def plotSignals(list_signals, num, **kwargs):

    percentage = kwargs.get('percentage', np.array([[None]]))
    if not(percentage):
        percentage = 1

    number_signals = len(list_signals)

    dimensions = []
    for signals in list_signals:
        dimensions.append(signals[0].dimension)
    max_dimension = max(dimensions)

    plt.figure(num=num, figsize=[10 * max_dimension, 5 * number_signals])

    for k in range(number_signals):
        for dim in range(dimensions[k]):
            time = np.linspace(0, list_signals[k][0].total_time, list_signals[k][0].number_steps)
            plt.subplot(number_signals, max_dimension, k * dimensions[k] + dim + 1)
            for signal in list_signals[k]:
                plt.plot(time[0:int(list_signals[k][0].number_steps * percentage) + 1], signal.data[dim, 0:int(list_signals[k][0].number_steps * percentage) + 1])
            plt.xlabel('Time [sec]')
            plt.ylabel('$y_{}'.format(dim + 1) + '$')

    plt.show()
