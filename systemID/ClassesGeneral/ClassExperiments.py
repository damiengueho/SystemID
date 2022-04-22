"""
Author: Damien GUEHO
Copyright: Copyright (C) 2022 Damien GUEHO
License: Public Domain
Version: 23
Date: April 2022
Python: 3.7.7
"""



import numpy as np

from systemID.ClassesGeneral.ClassSignal import DiscreteSignal, OutputSignal


class Experiments:
    def __init__(self, **kwargs):

        self.input_signals = kwargs.get('input_signals', [])
        self.output_signals = kwargs.get('output_signals', [])
        self.systems = kwargs.get('systems', [])

        self.number_experiments = len(self.output_signals)

        if len(self.input_signals) * len(self.systems) > 0 and len(self.output_signals) == 0:

            if self.input_signals[0].signal_type == 'Discrete':
                self.number_steps = self.input_signals[0].number_steps
                self.state_dimension = self.systems[0].state_dimension
                self.output_dimension = self.systems[0].output_dimension
                self.input_dimension = self.systems[0].input_dimension
                self.number_experiments = len(self.input_signals)
                self.frequency = self.input_signals[0].frequency
                self.total_time = self.input_signals[0].total_time
                self.output_signals = []
                for i in range(self.number_experiments):
                    self.output_signals.append(OutputSignal(self.input_signals[i], self.systems[i % len(self.systems)], **kwargs))

            if self.input_signals[0].signal_type == 'Continuous':
                self.state_dimension = self.systems[0].state_dimension
                self.output_dimension = self.systems[0].output_dimension
                self.input_dimension = self.systems[0].input_dimension
                self.number_experiments = len(self.input_signals)
                self.input_signals = []
                self.output_signals = []
                self.tspan = kwargs.get('tspan', np.array([0]))
                self.frequency = kwargs.get('frequency', 1)
                self.total_time = kwargs.get('total_time', 1)
                for i in range(self.number_experiments):
                    self.output_signals.append(OutputSignal(self.input_signals[i], self.systems[i % len(self.systems)], **kwargs))
                    self.input_signals.append(DiscreteSignal(self.input_dimension, self.total_time, self.frequency, signal_shape='External', data=self.input_signals[i].u(self.tspan)))
