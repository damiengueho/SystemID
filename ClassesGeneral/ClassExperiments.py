"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""



import numpy as np

from ClassesGeneral.ClassSignal import OutputSignal


class Experiments:
    def __init__(self, systems, input_signals, **kwargs):

        if input_signals[0].signal_type == 'Discrete':
            self.systems = systems
            self.input_signals = input_signals
            self.number_steps = input_signals[0].number_steps
            self.state_dimension = systems[0].state_dimension
            self.output_dimension = systems[0].output_dimension
            self.input_dimension = systems[0].input_dimension
            self.number_experiments = len(input_signals)
            self.frequency = systems[0].frequency
            self.output_signals = []
            for i in range(self.number_experiments):
                self.output_signals.append(OutputSignal(input_signals[i], systems[i], **kwargs))

        if input_signals[0].signal_type == 'Continuous':
            self.systems = systems
            self.input_signals = input_signals
            self.state_dimension = systems[0].state_dimension
            self.output_dimension = systems[0].output_dimension
            self.input_dimension = systems[0].input_dimension
            self.number_experiments = len(input_signals)
            self.output_signals = []
            self.tspan = kwargs.get('tspan', np.array([0, 0]))
            for i in range(self.number_experiments):
                self.output_signals.append(OutputSignal(input_signals[i], systems[i], **kwargs))
