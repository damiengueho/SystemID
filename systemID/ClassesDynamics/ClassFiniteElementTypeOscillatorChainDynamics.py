"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm, toeplitz
from scipy.interpolate import interp1d

from ClassesGeneral.ClassSignal import DiscreteSignal


class FiniteElementTypeOscillatorChainDynamics:
    def __init__(self, masses, spring_constants, damping_coefficients, nonlinear_damping_coefficients, **kwargs):
        self.dimension = len(masses)
        self.state_dimension = int(2 * self.dimension)
        self.input_dimension = self.dimension
        self.output_dimension = self.state_dimension
        self.masses = masses
        self.spring_constants = spring_constants
        self.damping_coefficients = damping_coefficients
        self.nonlinear_damping_coefficients = nonlinear_damping_coefficients

        self.M = np.diag(np.array(self.masses))
        if self.dimension == 1:
            self.L = np.array([[2]])
        else:
            self.L = toeplitz([2, -1] + [0] * (self.dimension - 2), [0, -1] + [0] * (self.dimension - 2))
        self.K = np.matmul(self.L, np.diag(np.array(self.spring_constants)))
        self.C = np.matmul(self.L, np.diag(np.array(self.damping_coefficients)))

        self.tspan = kwargs.get('tspan', np.array([0, 1, 2, 3]))
        self.nominal_x = kwargs.get('nominal_x', DiscreteSignal(self.state_dimension, 3, 1))
        self.nominal_x_interpolated = interp1d(self.tspan, self.nominal_x.data, 'cubic')
        self.nominal_u = kwargs.get('nominal_u', DiscreteSignal(self.input_dimension, 3, 1))
        self.nominal_u_interpolated = interp1d(self.tspan, self.nominal_u.data, 'cubic')
        self.dt = kwargs.get('dt', 0)

    def F(self, x, t, u):
        self.f3 = np.zeros(self.dimension)
        self.f3[0] = x[0] ** 3 - (x[1] - x[0]) ** 3
        for i in range(1, self.dimension - 2):
            self.f3[i] = (x[i] - x[i-1]) ** 3 - (x[i+1] - x[i]) ** 3
        self.f3[self.dimension - 1] = (x[self.dimension - 1] - x[self.dimension - 2]) ** 3 - x[self.dimension - 1] ** 3
        dxdt = np.zeros(self.state_dimension)
        dxdt[0:self.dimension] = x[self.dimension:]
        dxdt[self.dimension:] = -np.matmul(inv(self.M), self.K).dot(x[0:self.dimension]) - np.matmul(inv(self.M), self.C).dot(x[self.dimension:]) - self.nonlinear_damping_coefficients * self.f3 + u(t)
        return dxdt

    def G(self, x, t, u):
        return x

