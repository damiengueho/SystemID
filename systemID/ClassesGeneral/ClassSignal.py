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


from systemID.SystemIDAlgorithms.Propagation import propagation



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

        if self.signal_shape == 'Pulse':
            self.time_step = min(max(kwargs.get('time_step', 0), 0), self.number_steps - 1)
            self.magnitude_pulse = kwargs.get('magnitude_pulse', np.ones(self.dimension))
            self.data = np.zeros([self.dimension, self.number_steps])
            self.data[:, self.time_step] = self.magnitude_pulse
        elif self.signal_shape == 'Step':
            self.time_step = min(max(kwargs.get('time_step', 0), 0), self.number_steps - 1)
            self.magnitude_step = kwargs.get('magnitude_step', np.ones(self.dimension))
            self.data = np.zeros([self.dimension, self.number_steps])
            self.data[:, self.time_step:] = np.outer(self.magnitude_step, np.ones(self.number_steps - self.time_step))
        elif self.signal_shape == 'Ramp':
            self.time_step_start = min(max(kwargs.get('time_step_start', 0), 0), self.number_steps - 1)
            self.time_step_end = max(min(kwargs.get('time_step_end', self.number_steps - 1), self.number_steps - 1), 0)
            self.magnitude_max = kwargs.get('magnitude_max', np.ones(self.dimension))
            self.data = np.zeros([self.dimension, self.number_steps])
            self.data[:, self.time_step_start:self.time_step_end + 1] = np.linspace(np.zeros(self.dimension), self.magnitude_max, self.time_step_end - self.time_step_start + 1).T
            self.data[:, self.time_step_end + 1:] = np.outer(self.magnitude_max, np.ones(self.number_steps - self.time_step_end - 1))
        elif self.signal_shape == 'Triangle':
            self.magnitude_peak = kwargs.get('magnitude_peak', np.ones(self.dimension))
            self.data = np.zeros([self.dimension, self.number_steps])
            self.data[:, 0:int(self.number_steps / 2) + 1] = np.linspace(np.zeros(self.dimension), self.magnitude_peak, int(self.number_steps / 2) + 1).T
            self.data[:, int(self.number_steps / 2):self.number_steps] = np.linspace(self.magnitude_peak, np.zeros(self.dimension), self.number_steps - int(self.number_steps / 2)).T
        elif self.signal_shape == 'Sinusoid':
            self.magnitude_sinusoid = kwargs.get('magnitude_sinusoid', np.ones(self.dimension))
            self.frequency_sinusoid = kwargs.get('frequency_sinusoid', np.ones(self.dimension))
            self.phase_sinusoid = kwargs.get('phase_sinusoid', np.zeros(self.dimension))
            self.data = np.sin(2 * np.pi * np.outer(self.frequency_sinusoid, np.linspace(0, self.total_time, self.number_steps)) + np.outer(self.phase_sinusoid, np.ones(self.number_steps))) * self.magnitude_sinusoid[:, np.newaxis]
        elif self.signal_shape == 'White Noise':
            self.mean = kwargs.get('mean', np.zeros(self.dimension))
            self.covariance = kwargs.get('covariance', np.eye(self.dimension))
            self.data = np.matmul(LA.sqrtm(self.covariance), np.random.randn(self.dimension, self.number_steps)) + self.mean[:, np.newaxis]
        elif self.signal_shape == 'Half White Noise':
            self.mean = kwargs.get('mean', np.zeros(self.dimension))
            self.covariance = kwargs.get('covariance', np.eye(self.dimension))
            self.data = np.zeros([self.dimension, self.number_steps])
            self.data[:, 0:int(self.number_steps/2)] = np.matmul(LA.sqrtm(self.covariance), np.random.randn(self.dimension, int(self.number_steps/2))) + self.mean[:, np.newaxis]
            self.data[:, int(self.number_steps/2):] = np.outer(self.mean, np.ones([self.number_steps - int(self.number_steps/2)]))
        elif self.signal_shape == 'Combination':
            self.maximum_ramp = kwargs.get('maximum_ramp', np.ones(self.dimension))
            self.covariance = kwargs.get('covariance', np.eye(self.dimension))
            self.exponential_decay_rate = kwargs.get('exponential_decay_rate', -np.ones(self.dimension))
            self.data = np.zeros([self.dimension, self.number_steps])
            self.data[:, 0:int(self.number_steps / 3)] = np.linspace(np.zeros(self.dimension), self.maximum_ramp, int(self.number_steps / 3)).T
            self.data[:, int(self.number_steps / 3):2 * int(self.number_steps / 3)] = np.matmul(LA.sqrtm(self.covariance), np.random.randn(self.dimension, 2 * int(self.number_steps / 3) - int(self.number_steps / 3))) + self.maximum_ramp[:, np.newaxis]
            self.data[:, 2 * int(self.number_steps / 3):self.number_steps] = np.exp(np.outer(self.exponential_decay_rate, np.linspace(0, self.number_steps - 2 * int(self.number_steps / 3), self.number_steps - 2 * int(self.number_steps / 3)))) * self.maximum_ramp[:, np.newaxis]
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

            observer = kwargs.get('observer', False)
            reference_output_signal = kwargs.get('reference_output_signal', np.zeros([system.output_dimension, signal.number_steps]))

            process_noise = kwargs.get('process_noise', False)
            process_noise_signal = kwargs.get('process_noise_signal', DiscreteSignal(system.state_dimension, signal.total_time, signal.frequency))
            measurement_noise = kwargs.get('measurement_noise', False)
            measurement_noise_signal = kwargs.get('measurement_noise_signal', DiscreteSignal(system.output_dimension, signal.total_time, signal.frequency))

            data = propagation(signal, system, observer=observer, reference_output_signal=reference_output_signal, process_noise=process_noise, process_noise_signal=process_noise_signal, measurement_noise=measurement_noise, measurement_noise_signal=measurement_noise_signal)
            super().__init__(system.output_dimension, signal.total_time, signal.frequency, signal_shape='External', data=data[0])
            self.state = data[1]

        if signal.signal_type == 'Continuous':

            tspan = kwargs.get('tspan', np.array([0, 1]))
            total_time = tspan[-1]
            number_steps = len(tspan)
            frequency = int(round((number_steps - 1) / total_time))

            observer = kwargs.get('observer', False)
            reference_output_signal = kwargs.get('reference_output_signal', np.zeros([system.output_dimension, number_steps]))

            process_noise = kwargs.get('process_noise', False)
            process_noise_signal = kwargs.get('process_noise_signal', ContinuousSignal(system.state_dimension))
            measurement_noise = kwargs.get('measurement_noise', False)
            measurement_noise_signal = kwargs.get('measurement_noise_signal', ContinuousSignal(system.output_dimension))

            data = propagation(signal, system, tspan=tspan, observer=observer, reference_output_signal=reference_output_signal, process_noise=process_noise, process_noise_signal=process_noise_signal, measurement_noise=measurement_noise, measurement_noise_signal=measurement_noise_signal)
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
