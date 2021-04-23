"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 10
Date: April 2021
Python: 3.7.7
"""


from SystemIDAlgorithms.ObserverKalmanIdentificationAlgorithm import observerKalmanIdentificationAlgorithm
from SystemIDAlgorithms.ObserverKalmanIdentificationAlgorithmFull import observerKalmanIdentificationAlgorithmFull
from SystemIDAlgorithms.ObserverKalmanIdentificationAlgorithmObserver import observerKalmanIdentificationAlgorithmObserver
from SystemIDAlgorithms.ObserverKalmanIdentificationAlgorithmObserverWithInitialCondition import observerKalmanIdentificationAlgorithmObserverWithInitialCondition
from SystemIDAlgorithms.ObserverKalmanIdentificationAlgorithmObserverFull import observerKalmanIdentificationAlgorithmObserverFull
from SystemIDAlgorithms.GetMarkovParametersFromObserverMarkovParameters import getMarkovParametersFromObserverMarkovParameters
from SystemIDAlgorithms.GetObserverGainMarkovParametersFromObserverMarkovParameters import getObserverGainMarkovParametersFromObserverMarkovParameters
from SystemIDAlgorithms.GetTimeVaryingHankelMatrix import getTimeVaryingHankelMatrix


class OKIDObserver:
    def __init__(self, input_signal, output_signal, deadbeat_order, number_to_calculate):
        self.observer_markov_parameters, self.y, self.U = observerKalmanIdentificationAlgorithmObserver(input_signal, output_signal, deadbeat_order)
        self.markov_parameters = getMarkovParametersFromObserverMarkovParameters(self.observer_markov_parameters, number_to_calculate)
        self.observer_gain_markov_parameters = getObserverGainMarkovParametersFromObserverMarkovParameters(self.observer_markov_parameters, number_to_calculate)


class OKIDObserverFull:
    def __init__(self, input_signal, output_signal, number_to_calculate):
        self.observer_markov_parameters = observerKalmanIdentificationAlgorithmObserverFull(input_signal, output_signal)
        self.markov_parameters = getMarkovParametersFromObserverMarkovParameters(self.observer_markov_parameters, number_to_calculate)
        self.observer_gain_markov_parameters = getObserverGainMarkovParametersFromObserverMarkovParameters(self.observer_markov_parameters, number_to_calculate)


class OKIDObserverWithInitialCondition:
    def __init__(self, input_signal, output_signal, deadbeat_order, number_to_calculate):
        self.observer_markov_parameters, self.y, self.U = observerKalmanIdentificationAlgorithmObserverWithInitialCondition(input_signal, output_signal, deadbeat_order)
        self.markov_parameters = getMarkovParametersFromObserverMarkovParameters(self.observer_markov_parameters, number_to_calculate)
        self.observer_gain_markov_parameters = getObserverGainMarkovParametersFromObserverMarkovParameters(self.observer_markov_parameters, number_to_calculate)


class OKID:
    def __init__(self, input_signal, output_signal, deadbeat_order, number_to_calculate):
        self.markov_parameters = observerKalmanIdentificationAlgorithm(input_signal, output_signal, deadbeat_order)


class OKIDFull:
    def __init__(self, input_signal, output_signal):
        self.markov_parameters = observerKalmanIdentificationAlgorithmFull(input_signal, output_signal)


class TVOKIDObserver:
    def __init__(self, forced_experiments, free_decay_experiments, p, q, deadbeat_order):
        self.Y, self.hki, self.D, self.hki_observer1, self.hki_observer2, self.sv, self.E1, self.E2, self.E3, self.Vh = getTimeVaryingHankelMatrix(forced_experiments, free_decay_experiments, p, q, deadbeat_order)
