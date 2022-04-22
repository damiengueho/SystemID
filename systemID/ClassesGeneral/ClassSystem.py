"""
Author: Damien GUEHO
Copyright: Copyright (C) 2022 Damien GUEHO
License: Public Domain
Version: 23
Date: April 2022
Python: 3.7.7
"""


import numpy as np


from systemID.SystemIDAlgorithms.HigherOrderStateTransitionTensorsPropagation import higherOrderStateTransitionTensorsPropagation



class DiscreteLinearSystem:
    def __init__(self, frequency, x0, A, **kwargs):
        self.frequency = frequency
        self.dt = 1 / frequency
        self.system_type = 'Discrete Linear'
        self.x0 = x0
        self.A = A
        self.state_dimension = self.A(0).shape[0]
        self.B = kwargs.get('B', lambda t: np.zeros([self.state_dimension, 1]))
        self.input_dimension = self.B(0).shape[1]
        self.C = kwargs.get('C', lambda t: np.eye(self.state_dimension))
        self.output_dimension = self.C(0).shape[0]
        self.D = kwargs.get('D', lambda t: np.zeros([self.output_dimension, self.input_dimension]))
        self.K = kwargs.get('observer_gain', lambda t: np.zeros([self.state_dimension, self.output_dimension]))


class DiscreteNonlinearSystem:
    def __init__(self, frequency, x0, F, **kwargs):
        self.frequency = frequency
        self.dt = 1 / frequency
        self.system_type = 'Discrete Nonlinear'
        self.x0 = x0
        self.state_dimension = self.x0.shape[0]
        self.F = F
        self.input_dimension = kwargs.get('input_dimension', 1)
        self.G = kwargs.get('G', lambda x, t, u: x)
        self.output_dimension = self.G(np.zeros(self.state_dimension), 0, np.zeros(self.input_dimension)).shape[0]
        self.K = kwargs.get('observer_gain', lambda t: np.zeros([self.state_dimension, self.output_dimension]))


class ContinuousLinearSystem:
    def __init__(self, x0, A, **kwargs):
        self.system_type = 'Continuous Linear'
        self.x0 = x0
        self.A = A
        self.state_dimension = self.A(0).shape[0]
        self.B = kwargs.get('B', lambda t: np.zeros([self.state_dimension, 1]))
        self.input_dimension = self.B(0).shape[1]
        self.C = kwargs.get('C', lambda t: np.eye(self.state_dimension))
        self.output_dimension = self.C(0).shape[0]
        self.D = kwargs.get('D', lambda t: np.zeros([self.output_dimension, self.input_dimension]))
        self.K = kwargs.get('observer_gain', lambda t: np.zeros([self.state_dimension, self.output_dimension]))


class ContinuousBilinearSystem:
    def __init__(self, x0, A, **kwargs):
        self.system_type = 'Continuous Bilinear'
        self.x0 = x0
        self.A = A
        self.state_dimension = self.A(0).shape[0]
        self.B = kwargs.get('B', lambda t: np.zeros([self.state_dimension, 1]))
        self.input_dimension = self.B(0).shape[1]
        self.N = kwargs.get('N', lambda t: np.zeros([self.state_dimension, self.state_dimension * self.input_dimension]))
        self.C = kwargs.get('C', lambda t: np.eye(self.state_dimension))
        self.output_dimension = self.C(0).shape[0]
        self.D = kwargs.get('D', lambda t: np.zeros([self.output_dimension, self.input_dimension]))
        self.K = kwargs.get('observer_gain', lambda t: np.zeros([self.state_dimension, self.output_dimension]))


class ContinuousNonlinearSystem:
    def __init__(self, x0, F, **kwargs):
        self.system_type = 'Continuous Nonlinear'
        self.x0 = x0
        self.state_dimension = self.x0.shape[0]
        self.F = F
        self.input_dimension = kwargs.get('input_dimension', 1)
        self.G = kwargs.get('G', lambda x, t, u: x)
        self.output_dimension = self.G(np.zeros(self.state_dimension), 0, np.zeros(self.input_dimension)).shape[0]
        self.K = kwargs.get('observer_gain', lambda t: np.zeros([self.state_dimension, self.output_dimension]))


class ContinuousNonlinearSystemHigherOrderExpansion:
    def __init__(self, x0, F, G, Acs, x0_nominal, u_nominal, tspan):
        self.system_type = 'Continuous Nonlinear Higher Order Expansion'
        self.x0 = x0
        self.F = F
        self.G = G
        self.Acs = Acs
        self.x0_nominal = x0_nominal
        self.u_nominal = u_nominal
        self.tspan = tspan
        self.number_steps = len(self.tspan)

        self.order = len(self.Acs)

        self.A_vec = higherOrderStateTransitionTensorsPropagation(Acs, F, u_nominal, x0_nominal, tspan)
