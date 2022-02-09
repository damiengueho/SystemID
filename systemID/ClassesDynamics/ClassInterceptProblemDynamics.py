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

from ClassesGeneral.ClassSignal import DiscreteSignal
from SystemIDAlgorithms.HigherOrderStateTransitionTensorsPropagation import higherOrderStateTransitionTensorsPropagation
from SystemIDAlgorithms.Propagation import propagation
from ClassesGeneral.ClassSystem import ContinuousNonlinearSystem


class InterceptProblemDynamics:
    """
    Dynamics
    x' = Ax +Bu
    u* = -R^-1B^T(S(t)x(t)+v(t))
    S' = -A^TS(t)-S(t)A+S(t)BR^-!BS(t)-Q(t)
    S(tf) = Qf
    v' = -[A^T - S(t)BR^-1B^t]v(t) + Q(t)r(t)
    v(tf) = -Qfrf
    r(t) = 0
    rf = [xT yT 0 0]^T
    """

    def __init__(self, delta, alpha, beta, **kwargs):
        self.state_dimension = 4
        self.input_dimension = 2
        self.output_dimension = 4
        self.mu

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
        A = np.zeros([self.state_dimension, self.state_dimension])
        A[0, 2] = 1
        A[1, 3] = 1
        B = np.zeros([self.state_dimension, self.input_dimension])
        B[2, 0] = 1
        B[3, 1] = 1
        R = np.eye(self.input_dimension)





        dxdt = np.matmul(A, x) + np.matmul(B, u_opt)

        return dxdt

    def G(self, x, t, u):
        return x

    def Ac1(self, x, t, u):
        Ac1 = np.zeros([self.state_dimension, self.state_dimension])
        Ac1[0, 1] = 1
        Ac1[1, 0] = -self.alpha(t) - 3 * self.beta(t) * x[0] ** 2
        Ac1[1, 1] = -self.delta(t)
        return Ac1

    # def dPhi(self, Phi, t, x, u):
    #     return np.matmul(self.Ac1(x, t, u), Phi.reshape(2, 2)).reshape(4)
    #
    # def A(self, tk):
    #     A = odeint(self.dPhi, np.eye(2).reshape(4), np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
    #     return A[-1, :].reshape(2, 2)

    def Ac2(self, x, t, u):
        Ac2 = np.zeros([self.state_dimension, self.state_dimension, self.state_dimension])
        Ac2[1, 0, 0] = - 6 * self.beta(t) * x[0]
        return Ac2

    def Ac3(self, x, t, u):
        Ac3 = np.zeros([self.state_dimension, self.state_dimension, self.state_dimension, self.state_dimension])
        Ac3[1, 0, 0, 0] = - 6 * self.beta(t)
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