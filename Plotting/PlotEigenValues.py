"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA


def plotEigenValues(systems, num):


    plt.figure(num=num, figsize=[4, 4])
    for system in systems:
        plt.scatter(np.real(LA.eig(system.A(0))[0]), np.imag(LA.eig(system.A(0))[0]))
    plt.xlabel('Real value of eigen values')
    plt.ylabel('Imaginary value of eigen values')

    plt.show()


def plotHistoryEigenValues1System(system, number_steps, num):

    state_dimension = system.state_dimension
    dt = system.dt

    eig1 = np.zeros([number_steps, state_dimension])

    total_time = (number_steps - 1) * dt

    plt.figure(num=num, figsize=[6 * state_dimension, 8])
    time = np.linspace(0, total_time, number_steps)

    for i in range(number_steps):
        eig1[i, :] = np.real(LA.eig(system.A(i * dt))[0])

    eig1.sort(axis=1)

    for i in range(state_dimension):
        plt.subplot(state_dimension, 1, i + 1)
        plt.plot(time, np.transpose(eig1[:, i]), 'o')
        plt.xlabel('Time [sec]')
        plt.ylabel('Magnitude')

    plt.show()



def plotHistoryEigenValues2Systems(systems, number_steps, num):

    state_dimension = systems[0].state_dimension
    dt = systems[0].dt

    eig1 = np.zeros([number_steps, state_dimension])
    eig2 = np.zeros([number_steps, state_dimension])

    total_time = (number_steps - 1) * dt

    plt.figure(num=num, figsize=[6 * state_dimension, 4 * 2])
    time = np.linspace(0, total_time, number_steps)

    for i in range(number_steps):
        eig1[i, :] = np.abs(LA.eig(systems[0].A(i * dt))[0])
        eig2[i, :] = np.abs(LA.eig(systems[1].A(i * dt))[0])

    eig1.sort(axis=1)
    eig2.sort(axis=1)

    for i in range(state_dimension):
        plt.subplot(state_dimension, 2, 2 * i + 1)
        plt.plot(time, np.transpose(eig1[:, i]), 'o')
        plt.plot(time, np.transpose(eig2[:, i]), '*')
        plt.xlabel('Time [sec]')
        plt.ylabel('Magnitude')

    for i in range(state_dimension):
        plt.subplot(state_dimension, 2, 2 * i + 2)
        plt.plot(time, np.transpose(eig1[:, i]) - np.transpose(eig2[:, i]))
        plt.xlabel('Time [sec]')
        plt.ylabel('Error in Magnitude')

    plt.show()
