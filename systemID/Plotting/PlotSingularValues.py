"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""


import matplotlib.pyplot as plt
import numpy as np


def plotSingularValues(eras, names, num):

    number_eras = len(eras)

    sqrt_eras = np.sqrt(number_eras)

    cols = int(np.ceil(sqrt_eras))
    rows = int(np.ceil(number_eras / cols))

    plt.figure(num=num, figsize=[4 * cols, 4 * rows])
    for i in range(number_eras):
        plt.subplot(rows, cols, i + 1)
        plt.semilogy(np.diag(eras[i].Sigma[0:50, 0:50]), '.')
        plt.xlabel('Number of Singular Values')
        plt.ylabel('Magnitude of Singular Values')
        plt.title('Singular Value Decomposition of ' + names[i])

    plt.show()