"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""



import numpy as np
import scipy.linalg as LA


class TwoBodyProblemDynamics:
    def __init__(self, mu):
        self.state_dimension_cartesian = 6
        self.input_dimension_cartesian = 3
        self.output_dimension_cartesian = 3
        self.state_dimension_polar = 4
        self.input_dimension_polar = 2
        self.output_dimension_polar = 2
        self.mu = mu

    def F_cartesian(self, x, t, u):
        dxdt = np.zeros(6)
        r = LA.norm(x[0:3])
        dxdt[0] = x[3]
        dxdt[1] = x[4]
        dxdt[2] = x[5]
        dxdt[3] = -self.mu*x[0]/(r**3) + u(t)[0]
        dxdt[4] = -self.mu*x[1]/(r**3) + u(t)[1]
        dxdt[5] = -self.mu*x[2]/(r**3) + u(t)[2]
        return dxdt

    def G_cartesian(self, x, t, u):
        return np.array([x[0], x[1], x[2]])

    def F_polar(self, x, t, u):
        dxdt = np.zeros(4)
        dxdt[0] = x[2]
        dxdt[1] = x[3]
        dxdt[2] = -self.mu/x[0]**2 + x[0]*x[3]**2
        dxdt[3] = -2*x[2]*x[3]/x[0]
        return dxdt

    def G_polar(self, x, t, u):
        return np.array([x[0], x[1]])
