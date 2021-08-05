"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 12
Date: August 2021
Python: 3.7.7
"""


import numpy as np
import scipy.linalg as LA

from SystemIDAlgorithms.Propagation import propagation



class Signal:
    def __init__(self, dimension):
        self.dimension = dimension



class DiscreteSignal(Signal):
    def __init__(self, dimension, total_time, frequency, **kwargs):
        super().__init__(dimension)
        self.total_time = total_time
        self.frequency = frequency
        self.dt = 1/frequency
        self.number_steps = int(self.total_time * self.frequency) + 1
        self.signal_type = 'Discrete'
        self.signal_shape = kwargs.get('signal_shape', 'Zero')

        if self.signal_shape == 'White Noise':
            self.mean = kwargs.get('mean', np.zeros(self.dimension))
            self.covariance = kwargs.get('covariance', np.eye(self.dimension))
            self.data = np.matmul(LA.sqrtm(self.covariance), np.random.randn(self.dimension, self.number_steps)) + self.mean[:, np.newaxis]
        elif self.signal_shape == 'Pulse':
            self.magnitude_pulse = kwargs.get('magnitude_pulse', np.ones(self.dimension))
            self.data = np.zeros([self.dimension, self.number_steps])
            self.data[:, 0] = self.magnitude_pulse
        elif self.signal_shape == 'Sinusoid':
            self.magnitude_sinusoid = kwargs.get('magnitude_sinusoid', np.ones(self.dimension))
            self.frequency_sinusoid = kwargs.get('frequency_sinusoid', np.ones(self.dimension))
            self.phase_sinusoid = kwargs.get('phase_sinusoid', np.zeros(self.dimension))
            self.data = np.zeros([self.dimension, self.number_steps])
            for i in range(self.dimension):
                self.data[i, :] = self.magnitude_sinusoid[i] * np.sin(2 * np.pi * self.frequency_sinusoid[i] * np.linspace(0, self.total_time, self.number_steps) + self.phase_sinusoid[i])
        elif self.signal_shape == 'Triangle':
            self.magnitude_peak = kwargs.get('magnitude_peak', np.ones(self.dimension))
            self.data = np.zeros([self.dimension, self.number_steps])
            for i in range(self.dimension):
                self.data[i, 0:int(self.number_steps / 2) + 1] = np.linspace(0, self.magnitude_peak[i], int(self.number_steps / 2) + 1)
                self.data[i, int(self.number_steps / 2):self.number_steps] = np.linspace(self.magnitude_peak[i], 0, self.number_steps - int(self.number_steps / 2))
        elif self.signal_shape == 'Combination':
            self.maximum_ramp = kwargs.get('maximum_ramp', np.ones(self.dimension))
            self.exponential_decay_rate = kwargs.get('exponential_decay_rate', -np.ones(self.dimension))
            self.data = np.zeros([self.dimension, self.number_steps])
            for i in range(self.dimension):
                self.data[i, 0:int(self.number_steps / 3)] = np.linspace(0, self.maximum_ramp[i], int(self.number_steps / 3))
                self.data[i, int(self.number_steps / 3):2 * int(self.number_steps / 3)] = LA.sqrtm(self.covariance)[i, i] * np.random.randn(2 * int(self.number_steps / 3) - int(self.number_steps / 3)) + self.maximum_ramp[i]
                self.data[i, 2 * int(self.number_steps / 3):self.number_steps] = self.data[i, 2 * int(self.number_steps / 3) - 1] * np.exp(self.exponential_decay_rate[i] * np.linspace(0,self.number_steps - 2 * int(self.number_steps / 3),self.number_steps - 2 * int(self.number_steps / 3)))
        elif self.signal_shape == 'External':
            data = kwargs.get('data', np.zeros([self.dimension, self.number_steps]))
            if data.shape.__len__() == 1:
                data = data.reshape(-1, data.shape[0])
            self.data = data

        else:
            self.data = np.zeros([self.dimension, self.number_steps])




class ContinuousSignal(Signal):
    def __init__(self, dimension, **kwargs):
        super().__init__(dimension)
        self.signal_type = 'Continuous'
        self.signal_shape = kwargs.get('signal_shape', 'Zero')

        if self.signal_shape == 'White Noise':
            self.mean = kwargs.get('mean', np.zeros(self.dimension))
            self.covariance = kwargs.get('covariance', np.eye(self.dimension))
            def u(t):
                return np.matmul(LA.sqrtm(self.covariance), np.random.randn(self.dimension)) + self.mean
            self.u = u
        elif self.signal_shape == 'Pulse':
            self.magnitude_pulse = kwargs.get('magnitude_pulse', np.ones(self.dimension))
            def u(t):
                if t == 0:
                    out = self.magnitude_pulse
                else:
                    out = np.zeros(self.dimension)
                return out
            self.u = u
        elif self.signal_shape == 'Sinusoid':
            self.magnitude_sinusoid = kwargs.get('magnitude_sinusoid', np.ones(self.dimension))
            self.frequency_sinusoid = kwargs.get('frequency_sinusoid', np.ones(self.dimension))
            self.phase_sinusoid = kwargs.get('phase_sinusoid', np.zeros(self.dimension))
            if self.dimension == 1:
                def u(t):
                    return self.magnitude_sinusoid * np.sin(2 * np.pi * self.frequency_sinusoid * t + self.phase_sinusoid)
            else:
                def u(t):
                    data = self.magnitude_sinusoid.reshape((self.dimension, 1)) * np.sin(2 * np.pi * np.outer(self.frequency_sinusoid, t) + self.phase_sinusoid.reshape((self.dimension, 1)))
                    if data.shape[1] == 1:
                        return data[:, 0]
                    else:
                        return data
            self.u = u
        elif self.signal_shape == 'External':
            def zero(t):
                return np.zeros(self.dimension) * t
            self.u = kwargs.get('u', zero)
        else:
            def zero(t):
                return np.zeros(self.dimension) * t
            self.u = zero



class OutputSignal(DiscreteSignal):
    def __init__(self, signal, system, **kwargs):
        if signal.signal_type == 'Discrete':
            data = propagation(signal, system)
            super().__init__(system.output_dimension, signal.total_time, signal.frequency, signal_shape='External', data=data[0])
            self.state = data[1]
        if signal.signal_type == 'Continuous':
            tspan = kwargs.get('tspan', np.array([0, 1]))
            total_time = tspan[-1]
            number_steps = len(tspan)
            frequency = int(round((number_steps - 1) / total_time))
            data = propagation(signal, system, tspan=tspan)
            super().__init__(system.output_dimension, total_time, frequency, signal_shape='External', data=data[0])
            self.state = data[1]



def subtract2Signals(signal1, signal2):
    return DiscreteSignal(signal1.dimension, signal1.total_time, signal1.frequency, signal_shape='External', data=signal1.data-signal2.data)



def addSignals(signals):
    if len(signals) < 1:
        return []
    elif len(signals) == 1:
        return DiscreteSignal(signals[0].dimension, signals[0].total_time, signals[0].frequency, signal_shape='External', data=signals[0].data)
    else:
        data = signals[0].data
        for s in signals[1:]:
            data = data + s.data
        return DiscreteSignal(signals[0].dimension, signals[0].total_time, signals[0].frequency, signal_shape='External', data=data)



def stackSignals(signals):
    dimension = signals[0].dimension
    data = signals[0].data
    for signal in signals[1:]:
        dimension += signal.dimension
        data = np.concatenate((data, signal.data), axis=0)
    return DiscreteSignal(dimension, signals[0].total_time, signals[0].frequency, signal_shape='External', data=data)



def concatenateSignals(signals):
    return DiscreteSignal(signals[0].dimension, signals[0].total_time, signals[0].frequency, signal_shape='External', data=np.concatenate(signals, axis=1))
