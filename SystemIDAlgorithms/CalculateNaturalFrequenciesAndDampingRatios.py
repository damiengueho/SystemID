"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 16
Date: September 2021
Python: 3.7.7
"""


import numpy as np
from scipy import linalg as LA

from ClassesGeneral.ClassSignal import DiscreteSignal


def calculateNaturalFrequenciesAndDampingRatios(systems):

    natural_frequencies = []
    damping_ratios = []

    for system in systems:
        state_dimension = system.state_dimension
        dt = system.dt
        natural_frequencies.append(np.flip(np.imag(np.diag(LA.logm(np.diag(LA.eig(system.A(0))[0])) / dt)))[::2])
        damping_ratios.append(np.real(np.flip(np.diag(LA.logm(np.diag(LA.eig(system.A(0))[0])) / dt)))[::2])

    return natural_frequencies, damping_ratios
