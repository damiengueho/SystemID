"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


from systemID.SystemIDAlgorithms.ObserverKalmanIdentificationAlgorithm import observerKalmanIdentificationAlgorithm
from systemID.SystemIDAlgorithms.ObserverKalmanIdentificationAlgorithmWithObserver import observerKalmanIdentificationAlgorithmWithObserver
from systemID.SystemIDAlgorithms.GetMarkovParametersFromObserverMarkovParameters import getMarkovParametersFromObserverMarkovParameters
from systemID.SystemIDAlgorithms.GetObserverGainMarkovParametersFromObserverMarkovParameters import getObserverGainMarkovParametersFromObserverMarkovParameters
from systemID.SystemIDAlgorithms.GetMarkovParametersFromFrequencyResponseFunctions import getMarkovParametersFromFrequencyRepsonseFunctions
from systemID.SystemIDAlgorithms.TimeVaryingObserverKalmanIdentificationAlgorithmWithObserver import timeVaryingObserverKalmanIdentificationAlgorithmWithObserver


# class OKIDObserver:
#     def __init__(self, input_signal, output_signal, observer_order, number_to_calculate):
#         self.observer_markov_parameters, self.y, self.U = observerKalmanIdentificationAlgorithmObserver(input_signal, output_signal, observer_order)
#         self.markov_parameters = getMarkovParametersFromObserverMarkovParameters(self.observer_markov_parameters, number_to_calculate)
#         self.observer_gain_markov_parameters = getObserverGainMarkovParametersFromObserverMarkovParameters(self.observer_markov_parameters, number_to_calculate)
#
#
# class OKIDObserverFull:
#     def __init__(self, input_signal, output_signal, number_to_calculate):
#         self.observer_markov_parameters = observerKalmanIdentificationAlgorithmObserverFull(input_signal, output_signal)
#         self.markov_parameters = getMarkovParametersFromObserverMarkovParameters(self.observer_markov_parameters, number_to_calculate)
#         self.observer_gain_markov_parameters = getObserverGainMarkovParametersFromObserverMarkovParameters(self.observer_markov_parameters, number_to_calculate)

class OKID:
    def __init__(self, input_signal, output_signal, **kwargs):
        self.markov_parameters, self.U, self.y = observerKalmanIdentificationAlgorithm(input_signal, output_signal, **kwargs)


class OKIDWithObserver:
    def __init__(self, input_signal, output_signal, **kwargs):
        self.observer_markov_parameters, self.y, self.U = observerKalmanIdentificationAlgorithmWithObserver(input_signal, output_signal, **kwargs)
        self.markov_parameters = getMarkovParametersFromObserverMarkovParameters(self.observer_markov_parameters, **kwargs)
        self.observer_gain_markov_parameters = getObserverGainMarkovParametersFromObserverMarkovParameters(self.observer_markov_parameters, **kwargs)


class FrequencyResponseFunction:
    def __init__(self, experiments):
        self.DFT_u, self.DFT_y, self.Suu, self.Suy, self.Syu, self.Syy, self.Suu_averaged, self.Suy_averaged, self.Syu_averaged, self.Syy_averaged, self.transfer_function1, self.transfer_function2, self.markov_parameters1, self.markov_parameters2 = getMarkovParametersFromFrequencyRepsonseFunctions(experiments)


class TVOKIDWithObserver:
    def __init__(self, forced_experiments, **kwargs):
        self.D, self.hki, self.hkio = timeVaryingObserverKalmanIdentificationAlgorithmWithObserver(forced_experiments, **kwargs)
