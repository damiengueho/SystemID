"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""

import numpy as np

class continuous_linear_model:
    def __init__(self, x0, A, **kwargs):
        self.system_type = 'continuous linear'
        self.x0 = x0
        self.A = A
        self.state_dimension = self.A(0).shape[0]
        self.B = kwargs.get('B', lambda t: np.zeros([self.state_dimension, 1]))
        self.input_dimension = self.B(0).shape[1]
        self.C = kwargs.get('C', lambda t: np.eye(self.state_dimension))
        self.output_dimension = self.C(0).shape[0]
        self.D = kwargs.get('D', lambda t: np.zeros([self.output_dimension, self.input_dimension]))
        self.K = kwargs.get('observer_gain', lambda t: np.zeros([self.state_dimension, self.output_dimension]))


class continuous_bilinear_model:
    def __init__(self, x0, A, **kwargs):
        self.system_type = 'continuous bilinear'
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


class continuous_nonlinear_model:
    def __init__(self, x0, F, **kwargs):
        self.system_type = 'continuous nonlinear'
        self.x0 = x0
        self.state_dimension = self.x0.shape[0]
        self.F = F
        self.input_dimension = kwargs.get('input_dimension', 1)
        self.G = kwargs.get('G', lambda x, t, u: x)
        self.output_dimension = self.G(np.zeros(self.state_dimension), 0, np.zeros(self.input_dimension)).shape[0]
        self.K = kwargs.get('observer_gain', lambda t: np.zeros([self.state_dimension, self.output_dimension]))


class continuous_higher_order_model:
    def __init__(self, x0, F, **kwargs):
        self.system_type = 'continuous higher order'
        self.x0 = x0
        self.state_dimension = self.x0.shape[0]
        self.F = F
        self.input_dimension = kwargs.get('input_dimension', 1)
        self.G = kwargs.get('G', lambda x, t, u: x)
        self.output_dimension = self.G(np.zeros(self.state_dimension), 0, np.zeros(self.input_dimension)).shape[0]
        self.sensitivities = kwargs.get('sensitivities', [])
        self.x0_nominal = kwargs.get('x0_nominal', np.zeros(self.state_dimension))
        self.u_nominal = kwargs.get('u_nominal', lambda t: np.zeros(self.input_dimension))
        self.order = len(self.sensitivities)
