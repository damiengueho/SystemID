"""
Author: Damien GUEHO
Copyright: Copyright (C) 2022 Damien GUEHO
License: Public Domain
Version: 23
Date: April 2022
Python: 3.7.7
"""


import numpy as np
import scipy.linalg as LA


def calculateNaturalFrequenciesAndDampingRatios(systems):
    """
    Purpose:
        Compute the natural frequencies and associated damping ratios from the system matrix A of a list of systems.


    Parameters:
        - **systems** (``list``): list of systems.

    Returns:
        - **natural_frequencies** (``list``): list of natural frequencies.
        - **damping_ratios** (``list``): list of damping ratios.

    Imports:
        - ``import numpy as np``
        - ``import scipy.linalg as LA``

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
