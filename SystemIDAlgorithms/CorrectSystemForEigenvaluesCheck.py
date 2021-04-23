"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 10
Date: April 2021
Python: 3.7.7
"""



import numpy as np
import scipy.linalg as LA

from SystemIDAlgorithms.GetObservabilityMatrix import getObservabilityMatrix
from ClassesGeneral.ClassSystem import DiscreteLinearSystem


def correctSystemForEigenvaluesCheck(system, number_steps, p):

    # Dimension and parameters
    state_dimension, _ = system.A(0).shape
    dt = system.dt
    frequency = system.frequency

    # Initialize corrected A
    A_corrected_matrix = np.zeros([state_dimension, state_dimension, number_steps])

    # Apply correction
    for i in range(number_steps):
        O1 = getObservabilityMatrix(system.A, system.C, p, i * dt, dt)
        O2 = getObservabilityMatrix(system.A, system.C, p, (i + 1) * dt, dt)
        A_corrected_matrix[:, :, i] = np.matmul(LA.pinv(O1), np.matmul(O2, system.A(i * dt)))

    def A_corrected(tk):
        return A_corrected_matrix[:, :, int(round(tk * frequency))]

    return DiscreteLinearSystem(system.frequency, system.state_dimension, system.input_dimension, system.output_dimension, system.initial_states, system.name + 'corrected', A_corrected, system.B, system.C, system.D)
