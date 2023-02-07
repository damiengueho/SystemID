"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""


from systemID.functions.observer_kalman_identification_algorithm import observer_kalman_identification_algorithm
from systemID.functions.observer_kalman_identification_algorithm_with_observer import observer_kalman_identification_algorithm_with_observer
from systemID.functions.markov_parameters_from_observer_markov_parameters import markov_parameters_from_observer_markov_parameters
from systemID.functions.observer_gain_markov_parameters_from_observer_markov_parameters import observer_gain_markov_parameters_from_observer_markov_parameters
# from systemID.SystemIDAlgorithms.GetMarkovParametersFromFrequencyResponseFunctions import getMarkovParametersFromFrequencyResponseFunctions


class okid:
    def __init__(self, input_signals, output_signals, **kwargs):
        self.markov_parameters, self.U, self.y = observer_kalman_identification_algorithm(input_signals, output_signals, **kwargs)


class okid_with_observer:
    def __init__(self, input_signals, output_signals, **kwargs):
        self.observer_markov_parameters, self.U, self.y = observer_kalman_identification_algorithm_with_observer(input_signals, output_signals, **kwargs)
        self.markov_parameters = markov_parameters_from_observer_markov_parameters(self.observer_markov_parameters, **kwargs)
        self.observer_gain_markov_parameters = observer_gain_markov_parameters_from_observer_markov_parameters(self.observer_markov_parameters, **kwargs)


# class FrequencyResponseFunction:
#     def __init__(self, experiments):
#         self.DFT_u, self.DFT_y, self.Suu, self.Suy, self.Syu, self.Syy, self.Suu_averaged, self.Suy_averaged, self.Syu_averaged, self.Syy_averaged, self.transfer_function1, self.transfer_function2, self.markov_parameters1, self.markov_parameters2 = getMarkovParametersFromFrequencyResponseFunctions(experiments)

