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
from systemID.SystemIDAlgorithms.HigherOrderStateTransitionTensorsPropagation import higherOrderStateTransitionTensorsPropagation
from systemID.SystemIDAlgorithms.Propagation import propagation
from systemID.ClassesGeneral.ClassSystem import ContinuousNonlinearSystem


class FlutterAeroelasticWingDynamics:

    def __init__(self, mt, mw, Ia, a, rcg, xa, U, b, kh, ch, ca, rho, cla, clb, cma, cmb, ka0, ka1, ka2, ka3, ka4, **kwargs):
        self.state_dimension = 4
        self.input_dimension = 1
        self.output_dimension = 4
        self.mt = mt
        self.mw = mw
        self.Ia = Ia
        self.xa = xa
        self.U = U
        self.a = a
        self.b = b
        self.rcg = rcg
        self.kh = kh
        self.ch = ch
        self.ca = ca
        self.rho = rho
        self.cla = cla
        self.clb = clb
        self.cma = cma
        self.cmb = cmb
        self.ka0 = ka0
        self.ka1 = ka1
        self.ka2 = ka2
        self.ka3 = ka3
        self.ka4 = ka4
        self.d = self.mt * self.Ia - self.mw ** 2 * self.xa ** 2 * self.b ** 2
        self.k1 = self.Ia * self.kh / self.d
        self.k2 = (self.Ia * self.rho * self.b * self.cla + self.mw * self.xa * self.b ** 3 * self.rho * self.cma) / self.d
        self.k3 = - self.mw * self.xa * self.b * self.kh / self.d
        self.k4 = (- self.mw * self.xa * self.b ** 2 * self.rho * self.cla - self.mt * self.rho * self.b ** 2 * self.cma) / self.d
        self.c1 = (self.Ia * (self.ch + self.rho * self.U * self.b * self.cla) + self.mw * self.xa * self.rho * self.U * self.b ** 3 * self.cma) / self.d
        self.c2 = (self.Ia * self.rho * self.U * self.b ** 2 * self.cla * (1/2 - self.a) - self.mw * self.xa * self.b * self.ca + self.mw * self.xa + self.rho * self.U * self.b ** 4 * self.cma * (1/2 - self.a)) / self.d
        self.c3 = (-self.mw * self.xa * self.b * (self.ch + self.rho * self.U * self.b * self.cla) - self.mt * self.rho * self.U * self.b ** 2 * self.cma) / self.d
        self.c4 = (self.mt * (self.ca - self.rho * self.U * self.b ** 3 * self.cma * (1/2 - self.a)) - self.mw * self.xa * self.rho * self.U * self.b ** 3 * self.cla * (1/2 - self.a)) / self.d
        self.g3 = - (self.Ia * self.rho * self.b * self.clb + self.mw * self.xa * self.b ** 3 * self.rho * self.cmb) / self.d
        self.g4 = (self.mw * self.xa * self.rho * self.b **2 * self.clb + self.mt * self.rho * self.b ** 2 * self.cmb) / self.d


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


    def ka(self, alpha):
        return self.ka0 + self.ka1 * alpha + self.ka2 * alpha ** 2 + self.ka3 * alpha ** 3 + self.ka4 * alpha ** 4

    def dka1(self, alpha):
        return self.ka1 + 2 * self.ka2 * alpha + 3 * self.ka3 * alpha ** 2 + 4 * self.ka4 * alpha ** 3

    def dka2(self, alpha):
        return 2 * self.ka2 + 6 * self.ka3 * alpha + 12 * self.ka4 * alpha ** 2

    def dka3(self, alpha):
        return 6 * self.ka3 + 24 * self.ka4 * alpha

    def dka4(self, alpha):
        return 24 * self.ka4


    def p(self, x):
        return - self.mw * self.xa * self.b * self.ka(x) / self.d

    def dp1(self, x):
        return - self.mw * self.xa * self.b * self.dka1(x) / self.d

    def dp2(self, x):
        return - self.mw * self.xa * self.b * self.dka2(x) / self.d

    def dp3(self, x):
        return - self.mw * self.xa * self.b * self.dka3(x) / self.d

    def dp4(self, x):
        return - self.mw * self.xa * self.b * self.dka4(x) / self.d


    def q(self, x):
        return self.mt * self.ka(x) / self.d

    def dq1(self, x):
        return self.mt * self.dka1(x) / self.d

    def dq2(self, x):
        return self.mt * self.dka2(x) / self.d

    def dq3(self, x):
        return self.mt * self.dka3(x) / self.d

    def dq4(self, x):
        return self.mt * self.dka4(x) / self.d


    def F(self, x, t, u):
        dxdt = np.zeros(self.state_dimension)
        dxdt[0] = x[2]
        dxdt[1] = x[3]
        dxdt[2] = - self.k1 * x[0] - (self.k2 * self.U ** 2 + self.p(x[1])) * x[1] - self.c1 * x[2] - self.c2 * x[3]
        dxdt[3] = - self.k3 * x[0] - (self.k4 * self.U ** 2 + self.q(x[1])) * x[1] - self.c3 * x[2] - self.c4 * x[3]
        return dxdt


    def G(self, x, t, u):
        return x


    def Ac1(self, x, t, u):
        Ac1 = np.zeros([self.state_dimension, self.state_dimension])
        Ac1[0, 2] = 1
        Ac1[1, 3] = 1
        Ac1[2, 0] = - self.k1
        Ac1[2, 1] = - (self.k2 * self.U ** 2 + self.p(x[1])) - x[1] * self.dp1(x[1])
        Ac1[2, 2] = - self.c1
        Ac1[2, 3] = - self.c2
        Ac1[3, 0] = - self.k3
        Ac1[3, 1] = - (self.k4 * self.U ** 2 + self.q(x[1])) - x[1] * self.dq1(x[1])
        Ac1[3, 2] = - self.c3
        Ac1[3, 3] = - self.c4
        return Ac1


    def Ac2(self, x, t, u):
        Ac2 = np.zeros([self.state_dimension, self.state_dimension, self.state_dimension])
        Ac2[2, 1, 1] = - 2 * self.dp1(x[1]) - x[1] * self.dp2(x[1])
        Ac2[3, 1, 1] = - 2 * self.dq1(x[1]) - x[1] * self.dq2(x[1])
        return Ac2


    def Ac3(self, x, t, u):
        Ac3 = np.zeros([self.state_dimension, self.state_dimension, self.state_dimension, self.state_dimension])
        Ac3[2, 1, 1, 1] = - 3 * self.dp2(x[1]) - x[1] * self.dp3(x[1])
        Ac3[3, 1, 1, 1] = - 3 * self.dq2(x[1]) - x[1] * self.dq3(x[1])
        return Ac3


    def Ac4(self, x, t, u):
        Ac4 = np.zeros([self.state_dimension, self.state_dimension, self.state_dimension, self.state_dimension, self.state_dimension])
        Ac4[2, 1, 1, 1, 1] = - 4 * self.dp3(x[1]) - x[1] * self.dp4(x[1])
        Ac4[3, 1, 1, 1, 1] = - 4 * self.dq3(x[1]) - x[1] * self.dq4(x[1])
        return Ac4


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