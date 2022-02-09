"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""


import numpy as np
from scipy import linalg as LA


def calculateNaturalFrequenciesAndDampingRatios(systems):
    """
    Purpose:


    Parameters:
        -

    Returns:
        -

    Imports:
        -

    Description:


    See Also:
        -
    """

    natural_frequencies = []
    damping_ratios = []

    for system in systems:
        dt = system.dt
        all_natural_frequencies = np.imag(np.diag(LA.logm(np.diag(LA.eig(system.A(0))[0])) / dt))
        natural_frequencies.append(all_natural_frequencies[all_natural_frequencies >= 0])
        damping_ratios.append(np.real(np.flip(np.diag(LA.logm(np.diag(LA.eig(system.A(0))[0])) / dt)))[::2])

    return natural_frequencies, damping_ratios
