"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""


import numpy as np


from systemID.SystemIDAlgorithms.HigherOrderStateTransitionTensorsPropagation import higherOrderStateTransitionTensorsPropagation



class System:
    def __init__(self, state_dimension, input_dimension, output_dimension, initial_states, name):
        self.state_dimension = state_dimension
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.initial_states = initial_states
        self.x0 = self.initial_states[0][0]
        self.name = name


class DiscreteLinearSystem(System):
    def __init__(self, frequency, state_dimension, input_dimension, output_dimension, initial_states, name, A, B, C, D, **kwargs):
        super().__init__(state_dimension, input_dimension, output_dimension, initial_states, name)
        self.frequency = frequency
        self.dt = 1 / frequency
        self.system_type = 'Discrete Linear'
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.K = kwargs.get('observer_gain', np.zeros([self.state_dimension, self.output_dimension]))


class DiscreteNonlinearSystem(System):
    def __init__(self, frequency, state_dimension, input_dimension, output_dimension, initial_states, name, F, G, **kwargs):
        super().__init__(state_dimension, input_dimension, output_dimension, initial_states, name)
        self.frequency = frequency
        self.dt = 1 / frequency
        self.system_type = 'Discrete Nonlinear'
        self.F = F
        self.G = G
        self.K = kwargs.get('observer_gain', np.zeros([self.state_dimension, self.output_dimension]))


class ContinuousLinearSystem(System):
    def __init__(self, state_dimension, input_dimension, output_dimension, initial_states, name, A, B, C, D):
        super().__init__(state_dimension, input_dimension, output_dimension, initial_states, name)
        self.system_type = 'Continuous Linear'
        self.A = A
        self.B = B
        self.C = C
        self.D = D


class ContinuousBilinearSystem(System):
    def __init__(self, state_dimension, input_dimension, output_dimension, initial_states, name, A, N, B, C, D):
        super().__init__(state_dimension, input_dimension, output_dimension, initial_states, name)
        self.system_type = 'Continuous Bilinear'
        self.A = A
        self.N = N
        self.B = B
        self.C = C
        self.D = D


class ContinuousNonlinearSystem(System):
    def __init__(self, state_dimension, input_dimension, output_dimension, initial_states, name, F, G):
        super().__init__(state_dimension, input_dimension, output_dimension, initial_states, name)
        self.system_type = 'Continuous Nonlinear'
        self.F = F
        self.G = G


class ContinuousNonlinearSystemHigherOrderExpansion(System):
    def __init__(self, state_dimension, input_dimension, output_dimension, initial_states, name, F, G, Acs, x0_nominal, u_nominal, tspan):
        super().__init__(state_dimension, input_dimension, output_dimension, initial_states, name)
        self.system_type = 'Continuous Nonlinear Higher Order Expansion'
        self.F = F
        self.G = G
        self.Acs = Acs
        self.x0_nominal = x0_nominal
        self.u_nominal = u_nominal
        self.tspan = tspan
        self.number_steps = len(self.tspan)

        self.order = len(self.Acs)

        self.A_vec = higherOrderStateTransitionTensorsPropagation(Acs, F, u_nominal, x0_nominal, tspan)
