"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""



import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from ClassesGeneral.ClassSignal import DiscreteSignal


class VanDerPolOscillatorDynamics:
    def __init__(self, beta, mu, alpha, **kwargs):
        self.state_dimension = 2
        self.input_dimension = 1
        self.output_dimension = 2
        self.beta = beta
        self.mu = mu
        self.alpha = alpha
        self.tspan = kwargs.get('tspan', np.array([0, 1, 2, 3]))
        self.nominal_x = kwargs.get('nominal_x', DiscreteSignal(self.state_dimension, 'No nominal trajectory', 3, 1))
        self.nominal_x_interpolated = interp1d(self.tspan, self.nominal_x.data, 'cubic')
        self.nominal_u = kwargs.get('nominal_u', DiscreteSignal(self.input_dimension, 'No nominal input', 3, 1))
        self.nominal_u_interpolated = interp1d(self.tspan, self.nominal_u.data, 'cubic')
        self.dt = kwargs.get('dt', 0)

    def F(self, x, t, u):
        dxdt = np.zeros(2)
        dxdt[0] = x[1]
        dxdt[1] = self.beta(t) * x[0] + self.mu(t) * x[1] + self.alpha(t) * x[0]**2 * x[1] + u(t)
        return dxdt

    def G(self, x, t, u):
        return x

    def Ac(self, t):
        Ac = np.zeros([2, 2])
        Ac[0, 1] = 1
        Ac[1, 0] = self.beta + 2 * self.alpha * self.nominal_x_interpolated(t)[0] * self.nominal_x_interpolated(t)[1]
        Ac[1, 1] = self.mu + self.alpha * self.nominal_x_interpolated(t)[0] ** 2
        return Ac

    def dPhi(self, Phi, t):
        return np.matmul(self.Ac(t), Phi.reshape(2, 2)).reshape(4)

    def A(self, tk):
        A = odeint(self.dPhi, np.eye(2).reshape(4), np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return A[-1, :].reshape(2, 2)

    def Bc(self, t):
        Bc = np.zeros([2, 1])
        Bc[1, 0] = 1
        return Bc

    def dPsi(self, Psi, t):
        return np.matmul(self.Ac(t), Psi.reshape(2, 2)).reshape(4) + np.eye(2).reshape(4)

    def B(self, tk):
        B = odeint(self.dPsi, np.zeros([2, 2]).reshape(4), np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return np.matmul(B[-1, :].reshape(2, 2), self.Bc(tk))

    def C(self, tk):
        C = np.eye(2)
        return C

    def D(self, tk):
        D = np.zeros([2, 1])
        return D
