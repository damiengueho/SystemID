"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 10
Date: April 2021
Python: 3.7.7
"""


import numpy as np
from numpy import linalg as LA


def getMarkovParametersFull_Frequency(input_signal, output_signal):

    # Get data from Signals
    y = output_signal.data
    u = input_signal.data

    # Get dimensions
    input_dimension = input_signal.dimension
    number_steps = output_signal.number_steps

    # FFTs
    U = np.fft.fft(u) / number_steps
    Y = np.fft.fft(y) / number_steps


    # Spectral Densities
    Suu = np.array([np.diag(np.multiply(np.diag(U[0, :]), np.diag(np.transpose(U[0, :]))))])
    Suy = np.array([np.diag(np.multiply(np.diag(U[0, :]), np.diag(np.transpose(Y[0, :]))))])
    Syu = np.array([np.diag(np.multiply(np.diag(Y[0, :]), np.diag(np.transpose(U[0, :]))))])
    Syy = np.array([np.diag(np.multiply(np.diag(Y[0, :]), np.diag(np.transpose(Y[0, :]))))])


    # Transfer Functions G and G_tilde
    G = Syu / Suu
    G_tilde = Syy / Suy


    # Markov Parameters
    h = np.fft.ifft(G) * number_steps

    markov_parameters = [np.real(h[:, 0:input_dimension])]
    for i in range(number_steps - 1):
        markov_parameters.append(np.real(h[:, i * input_dimension + input_dimension:(i + 1) * input_dimension + input_dimension]))

    return markov_parameters
