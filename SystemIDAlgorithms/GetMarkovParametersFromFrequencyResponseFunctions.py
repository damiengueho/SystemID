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

from SystemIDAlgorithms.GetPowerOf2 import findPreviousPowerOf2


def getMarkovParametersFromFrequencyRepsonseFunctions(experiments):

    # Parameters
    number_experiments = experiments.number_experiments
    input_dimension = experiments.input_dimension
    output_dimension = experiments.output_dimension

    # Length of data to consider
    number_steps = findPreviousPowerOf2(experiments.number_steps)
    half_number_steps = number_steps

    # Individual DFTs
    U = np.zeros([input_dimension, half_number_steps, number_experiments], dtype=complex)
    Y = np.zeros([output_dimension, half_number_steps, number_experiments], dtype=complex)

    # Individual Spectral Densities
    Suu = np.zeros([input_dimension, input_dimension, half_number_steps, number_experiments], dtype=complex)
    Suy = np.zeros([input_dimension, output_dimension, half_number_steps, number_experiments], dtype=complex)
    Syu = np.zeros([output_dimension, input_dimension, half_number_steps, number_experiments], dtype=complex)
    Syy = np.zeros([output_dimension, output_dimension, half_number_steps, number_experiments], dtype=complex)

    # Averaged Spectral Densities
    Suu_averaged = np.zeros([input_dimension, input_dimension, half_number_steps], dtype=complex)
    Suy_averaged = np.zeros([input_dimension, output_dimension, half_number_steps], dtype=complex)
    Syu_averaged = np.zeros([output_dimension, input_dimension, half_number_steps], dtype=complex)
    Syy_averaged = np.zeros([output_dimension, output_dimension, half_number_steps], dtype=complex)

    for l in range(number_experiments):
        for i in range(input_dimension):
            U[i, :, l] = (np.fft.fft(experiments.input_signals[l].data[i, 0:number_steps]))[0: half_number_steps]
        for j in range(output_dimension):
            Y[j, :, l] = (np.fft.fft(experiments.output_signals[l].data[j, 0:number_steps]))[0: half_number_steps]
        for i1 in range(input_dimension):
            for i2 in range(input_dimension):
                Suu[i1, i2, :, l] = np.multiply(U[i1, :, l], np.conj(U[i2, :, l]))
        for i in range(input_dimension):
            for j in range(output_dimension):
                Suy[i, j, :, l] = np.multiply(U[i, :, l], np.conj(Y[j, :, l]))
                Syu[j, i, :, l] = np.multiply(Y[j, :, l], np.conj(U[i, :, l]))
        for j1 in range(output_dimension):
            for j2 in range(output_dimension):
                Syy[j1, j2, :, l] = np.multiply(Y[j1, :, l], np.conj(Y[j2, :, l]))

        Suu_averaged = Suu_averaged + Suu[:, :, :, l] / number_experiments
        Suy_averaged = Suy_averaged + Suy[:, :, :, l] / number_experiments
        Syu_averaged = Syu_averaged + Syu[:, :, :, l] / number_experiments
        Syy_averaged = Syy_averaged + Syy[:, :, :, l] / number_experiments

    # Calculate Transfer Functions
    transfer_function1 = np.zeros([output_dimension, input_dimension, half_number_steps], dtype=complex)
    transfer_function2 = np.zeros([output_dimension, input_dimension, half_number_steps], dtype=complex)
    for k in range(half_number_steps):
        transfer_function1[:, :, k] = np.matmul(Syu_averaged[:, :, k], LA.pinv(np.conj(Suu_averaged[:, :, k])))
        transfer_function2[:, :, k] = np.matmul(Syy_averaged[:, :, k], LA.pinv(np.conj(Suy_averaged[:, :, k])))

    # Compute Markov Parameters
    h1 = np.zeros([output_dimension, input_dimension, number_steps])
    h2 = np.zeros([output_dimension, input_dimension, number_steps])
    for i in range(input_dimension):
        for j in range(output_dimension):
            h1[j, i, :] = np.real(np.fft.ifft(transfer_function1[j, i, :]))
            h2[j, i, :] = np.real(np.fft.ifft(transfer_function2[j, i, :]))

    # Create Markov Parameters lists
    markov_parameters1 = []
    markov_parameters2 = []
    for k in range(number_steps):
        markov_parameters1.append(h1[:, :, k])
        markov_parameters2.append(h2[:, :, k])

    return U, Y, Suu, Suy, Syu, Syy, Suu_averaged, Suy_averaged, Syu_averaged, Syy_averaged, transfer_function1, transfer_function2, h1, h2, markov_parameters1, markov_parameters2
