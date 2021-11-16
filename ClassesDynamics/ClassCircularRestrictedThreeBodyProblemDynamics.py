"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""



import numpy as np
import scipy.linalg as LA

from ClassesGeneral.ClassSignal import DiscreteSignal
from SystemIDAlgorithms.HigherOrderStateTransitionTensorsPropagation import higherOrderStateTransitionTensorsPropagation
from SystemIDAlgorithms.Propagation import propagation
from ClassesGeneral.ClassSystem import ContinuousNonlinearSystem


class CircularRestricedThreeBodyProblemDynamics:
    def __init__(self, mu, **kwargs):
        self.state_dimension = 6
        self.input_dimension = 3
        self.output_dimension = 6
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
        r1 = LA.norm(np.array([x[0] + self.mu(t), x[1], x[2]]))
        r2 = LA.norm(np.array([x[0] + self.mu(t) - 1, x[1], x[2]]))
        dxdt[0] = x[3]
        dxdt[1] = x[4]
        dxdt[2] = x[5]
        dxdt[3] = 2 * x[4] + x[0] - (1 - self.mu(t)) * (x[0] + self.mu(t)) / (r1 ** 3) - self.mu(t) * (x[0] - 1 + self.mu(t)) / (r2 ** 3)
        dxdt[4] = -2 * x[3] + x[1] - (1 - self.mu(t)) * x[1] / (r1 ** 3) - self.mu(t) * x[1] / (r2 ** 3)
        dxdt[5] = - (1 - self.mu(t)) * x[2] / (r1 ** 3) - self.mu(t) * x[2] / (r2 ** 3)
        return dxdt

    def G(self, x, t, u):
        return x

    def Ac1(self, x, t, u):
        Ac1 = np.zeros([self.state_dimension, self.state_dimension])
        Ac1[0, 3] = 1
        Ac1[1, 4] = 1
        Ac1[2, 5] = 1
        Ac1[3, 0] = (self.mu(t) - 1)/(abs(x[0] + self.mu(t))**2 + abs(x[1])**2 + abs(x[2])**2)**(3/2) - self.mu(t)/(abs(x[0] + self.mu(t) - 1)**2 + abs(x[1])**2 + abs(x[2])**2)**(3/2) - (3*abs(x[0] + self.mu(t))*np.sign(x[0] + self.mu(t))*(x[0] + self.mu(t))*(self.mu(t) - 1))/(abs(x[0] + self.mu(t))**2 + abs(x[1])**2 + abs(x[2])**2)**(5/2) + (3*self.mu(t)*abs(x[0] + self.mu(t) - 1)*np.sign(x[0] + self.mu(t) - 1)*(x[0] + self.mu(t) - 1))/(abs(x[0] + self.mu(t) - 1)**2 + abs(x[1])**2 + abs(x[2])**2)**(5/2) + 1
        Ac1[3, 1] = (3*self.mu(t)*abs(x[1])*np.sign(x[1])*(x[0] + self.mu(t) - 1))/(abs(x[0] + self.mu(t) - 1)**2 + abs(x[1])**2 + abs(x[2])**2)**(5/2) - (3*abs(x[1])*np.sign(x[1])*(x[0] + self.mu(t))*(self.mu(t) - 1))/(abs(x[0] + self.mu(t))**2 + abs(x[1])**2 + abs(x[2])**2)**(5/2)
        Ac1[3, 2] = (3*self.mu(t)*abs(x[2])*np.sign(x[2])*(x[0] + self.mu(t) - 1))/(abs(x[0] + self.mu(t) - 1)**2 + abs(x[1])**2 + abs(x[2])**2)**(5/2) - (3*abs(x[2])*np.sign(x[2])*(x[0] + self.mu(t))*(self.mu(t) - 1))/(abs(x[0] + self.mu(t))**2 + abs(x[1])**2 + abs(x[2])**2)**(5/2)
        Ac1[3, 4] = 2
        Ac1[4, 0] = (3*x[1]*self.mu(t)*abs(x[0] + self.mu(t) - 1)*np.sign(x[0] + self.mu(t) - 1))/(abs(x[0] + self.mu(t) - 1)**2 + abs(x[1])**2 + abs(x[2])**2)**(5/2) - (3*x[1]*abs(x[0] + self.mu(t))*np.sign(x[0] + self.mu(t))*(self.mu(t) - 1))/(abs(x[0] + self.mu(t))**2 + abs(x[1])**2 + abs(x[2])**2)**(5/2)
        Ac1[4, 1] = (self.mu(t) - 1)/(abs(x[0] + self.mu(t))**2 + abs(x[1])**2 + abs(x[2])**2)**(3/2) - self.mu(t)/(abs(x[0] + self.mu(t) - 1)**2 + abs(x[1])**2 + abs(x[2])**2)**(3/2) + (3*x[1]*self.mu(t)*abs(x[1])*np.sign(x[1]))/(abs(x[0] + self.mu(t) - 1)**2 + abs(x[1])**2 + abs(x[2])**2)**(5/2) - (3*x[1]*abs(x[1])*np.sign(x[1])*(self.mu(t) - 1))/(abs(x[0] + self.mu(t))**2 + abs(x[1])**2 + abs(x[2])**2)**(5/2) + 1
        Ac1[4, 2] = (3*x[1]*self.mu(t)*abs(x[2])*np.sign(x[2]))/(abs(x[0] + self.mu(t) - 1)**2 + abs(x[1])**2 + abs(x[2])**2)**(5/2) - (3*x[1]*abs(x[2])*np.sign(x[2])*(self.mu(t) - 1))/(abs(x[0] + self.mu(t))**2 + abs(x[1])**2 + abs(x[2])**2)**(5/2)
        Ac1[4, 3] = -2
        Ac1[5, 0] = (3*x[2]*self.mu(t)*abs(x[0] + self.mu(t) - 1)*np.sign(x[0] + self.mu(t) - 1))/(abs(x[0] + self.mu(t) - 1)**2 + abs(x[1])**2 + abs(x[2])**2)**(5/2) - (3*x[2]*abs(x[0] + self.mu(t))*np.sign(x[0] + self.mu(t))*(self.mu(t) - 1))/(abs(x[0] + self.mu(t))**2 + abs(x[1])**2 + abs(x[2])**2)**(5/2)
        Ac1[5, 1] = (3*x[2]*self.mu(t)*abs(x[1])*np.sign(x[1]))/(abs(x[0] + self.mu(t) - 1)**2 + abs(x[1])**2 + abs(x[2])**2)**(5/2) - (3*x[2]*abs(x[1])*np.sign(x[1])*(self.mu(t) - 1))/(abs(x[0] + self.mu(t))**2 + abs(x[1])**2 + abs(x[2])**2)**(5/2)
        Ac1[5, 2] = (self.mu(t) - 1)/(abs(x[0] + self.mu(t))**2 + abs(x[1])**2 + abs(x[2])**2)**(3/2) - self.mu(t)/(abs(x[0] + self.mu(t) - 1)**2 + abs(x[1])**2 + abs(x[2])**2)**(3/2) + (3*x[2]*self.mu(t)*abs(x[2])*np.sign(x[2]))/(abs(x[0] + self.mu(t) - 1)**2 + abs(x[1])**2 + abs(x[2])**2)**(5/2) - (3*x[2]*abs(x[2])*np.sign(x[2])*(self.mu(t) - 1))/(abs(x[0] + self.mu(t))**2 + abs(x[1])**2 + abs(x[2])**2)**(5/2)


        return Ac1

    def Ac2(self, x, t, u):
        Ac2 = np.zeros([self.state_dimension, self.state_dimension, self.state_dimension])
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
