"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""

import numpy as np



class discrete_signal:
    def __init__(self, **kwargs):
        self.signal_type = 'discrete'

        self.data = kwargs.get('data', np.array([0]))
        if not(isinstance(self.data, np.ndarray)):
            raise ValueError("Data needs to be a numpy.ndarray of dimension (L,) or (m, L)")
        if not(1 <= len(self.data.shape) <= 2):
            raise ValueError("Data needs to be a numpy.ndarray of dimension (L,) or (m, L)")
        if len(self.data.shape) == 1:
            self.dimension = 1
            self.number_steps = int(self.data.shape[0])
            self.data = np.expand_dims(self.data, axis=0)
        else:
            self.dimension = int(self.data.shape[0])
            self.number_steps = int(self.data.shape[1])

        self.frequency = kwargs.get('frequency', 1)
        if not(isinstance(self.frequency, float) or isinstance(self.frequency, int)):
            raise ValueError("Frequency needs to be a positive integer or float.")
        if not(self.frequency > 0):
            raise ValueError("Frequency needs to be a positive integer or float.")

        self.total_time = float((self.number_steps - 1) / self.frequency)

# def normalize_signal(discrete_signal):
#