"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 12
Date: August 2021
Python: 3.7.7
"""


class System:
    def __init__(self, state_dimension, input_dimension, output_dimension, initial_states, name):
        self.state_dimension = state_dimension
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.initial_states = initial_states
        self.x0 = self.initial_states[0][0]
        self.name = name


class DiscreteLinearSystem(System):
    def __init__(self, frequency, state_dimension, input_dimension, output_dimension, initial_states, name, A, B, C, D):
        super().__init__(state_dimension, input_dimension, output_dimension, initial_states, name)
        self.frequency = frequency
        self.dt = 1 / frequency
        self.system_type = 'Discrete Linear'
        self.A = A
        self.B = B
        self.C = C
        self.D = D


class DiscreteNonlinearSystem(System):
    def __init__(self, frequency, state_dimension, input_dimension, output_dimension, initial_states, name, F, G):
        super().__init__(state_dimension, input_dimension, output_dimension, initial_states, name)
        self.frequency = frequency
        self.dt = 1 / frequency
        self.system_type = 'Discrete Nonlinear'
        self.F = F
        self.G = G


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
