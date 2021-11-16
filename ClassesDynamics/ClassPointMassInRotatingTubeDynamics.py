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


class PointMassInRotatingTubeDynamics:
    def __init__(self, dt, mass, spring_constant, theta_dot):
        self.state_dimension = 2
        self.input_dimension = 1
        self.output_dimension = 2
        self.dt = dt
        self.mass = mass
        self.spring_constant = spring_constant
        self.theta_dot = theta_dot

    def Ac(self, t):
        Ac = np.zeros([2, 2])
        Ac[0, 1] = 1
        Ac[1, 0] = self.theta_dot(t) ** 2 - self.spring_constant / self.mass
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
        D[0, 0] = 0
        D[1, 0] = 1
        return D
