"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 17
Date: October 2021
Python: 3.7.7
"""



import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from ClassesGeneral.ClassSignal import DiscreteSignal


class LorenzSystemDynamics:
    def __init__(self, sigma, rho, beta, **kwargs):
        self.state_dimension = 3
        self.input_dimension = 3
        self.output_dimension = 3
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.tspan = kwargs.get('tspan', np.array([0, 1, 2, 3]))
        self.nominal_x = kwargs.get('nominal_x', DiscreteSignal(self.state_dimension, 'No nominal trajectory', 3, 1))
        self.nominal_x_interpolated = interp1d(self.tspan, self.nominal_x.data, 'cubic')
        self.nominal_u = kwargs.get('nominal_u', DiscreteSignal(self.input_dimension, 'No nominal input', 3, 1))
        self.nominal_u_interpolated = interp1d(self.tspan, self.nominal_u.data, 'cubic')
        self.dt = kwargs.get('dt', 0)

    def F(self, x, t, u):
        dxdt = np.zeros(3)
        dxdt[0] = self.sigma * (x[1] - x[0]) + u(t)[0]
        dxdt[1] = x[0] * (self.rho - x[2]) - x[1] + u(t)[1]
        dxdt[2] = x[0] * x[1] - self.beta * x[2] + u(t)[2]
        return dxdt

    def G(self, x, t, u):
        return np.array([x[0], x[1], x[2]])

    def Ac(self, t):
        Ac = np.zeros([3, 3])
        Ac[0, 0] = -self.sigma
        Ac[0, 1] = self.sigma
        Ac[1, 0] = self.rho - self.nominal_x_interpolated(t)[2]
        Ac[1, 1] = -1
        Ac[1, 2] = -self.nominal_x_interpolated(t)[0]
        Ac[2, 0] = self.nominal_x_interpolated(t)[1]
        Ac[2, 1] = self.nominal_x_interpolated(t)[0]
        Ac[2, 2] = -self.beta
        return Ac

    def dPhi(self, Phi, t):
        return np.matmul(self.Ac(t), Phi.reshape(3, 3)).reshape(9)

    def A(self, tk):
        A = odeint(self.dPhi, np.eye(3).reshape(9), np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return A[-1, :].reshape(3, 3)

    def Bc(self, t):
        Bc = np.eye(3)
        return Bc

    def dPsi(self, Psi, t):
        return np.matmul(self.Ac(t), Psi.reshape(3, 3)).reshape(9) + np.eye(3).reshape(9)

    def B(self, tk):
        B = odeint(self.dPsi, np.zeros([3, 3]).reshape(9), np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return np.matmul(B[-1, :].reshape(3, 3), self.Bc(tk))

    def C(self, tk):
        C = np.eye(3)
        return C

    def D(self, tk):
        D = np.zeros([3, 3])
        return D
