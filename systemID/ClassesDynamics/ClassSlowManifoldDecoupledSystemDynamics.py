"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""


import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from systemID.ClassesGeneral.ClassSignal import DiscreteSignal


class SlowManifoldDecoupledSystemDynamics:
    def __init__(self, mu, l, **kwargs):
        self.state_dimension = 2
        self.input_dimension = 2
        self.output_dimension = 2
        self.mu = mu
        self.l = l
        self.tspan = kwargs.get('tspan', np.array([0, 1, 2, 3]))
        self.nominal_x = kwargs.get('nominal_x', DiscreteSignal(self.state_dimension, 3, 1))
        self.nominal_x_interpolated = interp1d(self.tspan, self.nominal_x.data, 'cubic')
        self.nominal_u = kwargs.get('nominal_u', DiscreteSignal(self.input_dimension, 3, 1))
        self.nominal_u_interpolated = interp1d(self.tspan, self.nominal_u.data, 'cubic')
        self.dt = kwargs.get('dt', 0)

    def F(self, x, t, u):
        dxdt = np.zeros(self.state_dimension)
        dxdt[0] = self.mu(t) * x[0] + u(t)[0]
        dxdt[1] = self.l(t) * (x[1] - x[0] ** 2) + u(t)[1]
        return dxdt

    def G(self, x, t, u):
        return x

    def Ac(self, t):
        Ac = np.zeros([self.state_dimension, self.state_dimension])
        Ac[0, 0] = self.mu(t)
        Ac[1, 1] = self.l(t)
        Ac[1, 0] = -2 * self.l(t) * self.nominal_x_interpolated(t)[0]
        return Ac

    def dPhi(self, Phi, t):
        return np.matmul(self.Ac(t), Phi.reshape(self.state_dimension, self.state_dimension)).reshape(self.state_dimension ** 2)

    def A(self, tk):
        A = odeint(self.dPhi, np.eye(self.state_dimension).reshape(self.state_dimension ** 2), np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return A[-1, :].reshape(self.state_dimension, self.state_dimension)

    def Bc(self, t):
        Bc = np.zeros([self.state_dimension, self.input_dimension])
        Bc[0, 0] = 1
        Bc[1, 1] = 1
        return Bc

    def dPsi(self, Psi, t):
        return np.matmul(self.Ac(t), Psi.reshape(3, 3)).reshape(9) + np.eye(3).reshape(9)

    def B(self, tk):
        B = odeint(self.dPsi, np.zeros([3, 3]).reshape(9), np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return np.matmul(B[-1, :].reshape(3, 3), self.Bc(tk))

    def C(self, tk):
        C = np.zeros([self.output_dimension, self.state_dimension])
        C[0, 0] = 1
        C[1, 1] = 1
        return C

    def D(self, tk):
        D = np.zeros([self.output_dimension, self.input_dimension])
        return D
