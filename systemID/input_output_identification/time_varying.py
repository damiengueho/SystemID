"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""


from systemID.functions.time_varying_observer_kalman_identification_algorithm_with_observer import time_varying_observer_kalman_identification_algorithm_with_observer



class tvokid:
    def __init__(self, input_signals, output_signals, **kwargs):
        self.D, self.hki, self.hkio, self.hki_observer1, self.hki_observer2 = time_varying_observer_kalman_identification_algorithm_with_observer(input_signals, output_signals, **kwargs)
