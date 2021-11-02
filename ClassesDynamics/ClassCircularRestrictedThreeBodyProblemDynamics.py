"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 18
Date: October 2021
Python: 3.7.7
"""



import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from ClassesGeneral.ClassSignal import DiscreteSignal
from SystemIDAlgorithms.HigherOrderStateTransitionTensorsPropagation import higherOrderStateTransitionTensorsPropagation
from SystemIDAlgorithms.Propagation import propagation
from ClassesGeneral.ClassSystem import ContinuousNonlinearSystem


class CircularRestricedThreeBodyProblemDynamics:
    def __init__(self, omega, mu, **kwargs):
        self.state_dimension = 6
        self.input_dimension = 3
        self.output_dimension = 6
        self.omega = omega
        self.mu = mu

        self.nominal = kwargs.get('nominal', False)
        if self.nominal:
            self.initial_states = kwargs.get('initial_states', np.zeros(self.state_dimension))

            def zero(t):
                return np.zeros(self.input_dimension)

            self.nominal_u = kwargs.get('nominal_u', zero)
            self.nominal_system = ContinuousNonlinearSystem(self.state_dimension, self.input_dimension, self.output_dimension, self.initial_states, 'Nominal System', self.F, self.G)
            self.dt = kwargs.get('dt', 0)
            self.tspan = kwargs.get('tspan', np.array([0, 1]))
            _, self.nominal_x = propagation(self.nominal_u, self.nominal_system, tspan=self.tspan)


    def F(self, x, t, u):
        dxdt = np.zeros(self.state_dimension)
        r1 = np.norm(np.array([x[0] + self.mu(t), x[1], x[2]]))
        r2 = np.norm(np.array([x[0] + self.mu(t) - 1, x[1], x[2]]))
        dxdt[0] = x[3]
        dxdt[1] = x[4]
        dxdt[2] = x[5]
        dxdt[3] = 2 * x[4] * self.omega(t) + x[0] * self.omega(t) ** 2 - (1 - self.mu(t)) * (x[0] + self.mu(t)) / (r1 ** 3) - self.mu(t) * (x[0] - 1 + self.mu(t)) / (r2 ** 3)
        dxdt[4] = -2 * x[3] * self.omega(t) + x[1] * self.omega(t) ** 2 - (1 - self.mu(t)) * x[1] / (r1 ** 3) - self.mu(t) * x[1] / (r2 ** 3)
        dxdt[5] = - (1 - self.mu(t)) * x[2] / (r1 ** 3) - self.mu(t) * x[2] / (r2 ** 3)
        return dxdt

    def G(self, x, t, u):
        return x

    def Ac1(self, x, t, u):
        Ac1 = np.zeros([self.state_dimension, self.state_dimension])
        Ac1[0, 0] = -self.sigma(t)
        Ac1[0, 1] = self.sigma(t)
        Ac1[1, 0] = self.rho(t)
        Ac1[1, 1] = -1
        Ac1[1, 2] = -x[0]
        Ac1[2, 0] = x[1]
        Ac1[2, 1] = x[0]
        Ac1[2, 2] = -self.beta(t)
        return Ac1

    def Ac2(self, x, t, u):
        Ac2 = np.zeros([self.state_dimension, self.state_dimension, self.state_dimension])
        Ac2[1, 2, 0] = -1
        Ac2[2, 0, 1] = 1
        Ac2[2, 1, 0] = 1
        return Ac2

    def Ac3(self, x, t, u):
        Ac3 = np.zeros([self.state_dimension, self.state_dimension, self.state_dimension, self.state_dimension])
        return Ac3

    def Ac4(self, x, t, u):
        Ac4 = np.zeros([self.state_dimension, self.state_dimension, self.state_dimension, self.state_dimension, self.state_dimension])
        return Ac4

    ## Do B

    # def Bc(self, t):
    #     Bc = np.zeros([self.state_dimension, self.input_dimension])
    #     Bc[1, 0] = 1
    #     return Bc
    #
    # def dPsi(self, Psi, x, t, u):
    #     return np.matmul(self.Ac1(x, t, u), Psi.reshape(self.state_dimension, self.state_dimension)).reshape(self.state_dimension**2) + np.eye(self.state_dimension).reshape(self.state_dimension**2)

    def B(self, tk):
        # B = odeint(self.dPsi, np.zeros([self.state_dimension, self.state_dimension]).reshape(self.state_dimension**2), np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        # return np.matmul(B[-1, :].reshape(self.state_dimension, self.state_dimension), self.Bc(tk))
        return np.zeros([self.state_dimension, self.input_dimension])

    def C(self, tk):
        C = np.eye(self.state_dimension)
        return C

    def D(self, tk):
        D = np.zeros([self.output_dimension, self.input_dimension])
        return D
