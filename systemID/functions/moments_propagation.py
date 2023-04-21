"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""


import numpy as np


def moments_propagation(initial_moments, state_transition_tensors):
    """
        Purpose:
            This program computes the time evolution of the moments of a probability density function, given its initial moments
            and the time evolution of higher-order state transition tensors.

        Parameters:
            - **initial_moments** (`list`): the initial moments for a given probability density function.
            -

        Returns:
            -

        Imports:
            -

        Description:


        See Also:
            -
    """

    ## Order of the expansion and dimension of the system
    order = len(initial_moments)
    state_dimension = initial_moments[0].shape[0]
    number_steps = state_transition_tensors[0].shape[-1]


    ## Initialization
    P1 = np.zeros([state_dimension, number_steps])
    P1[:, 0] = initial_moments[0]
    Phi1 = state_transition_tensors[0]

    if order > 1:
        P2 = np.zeros([state_dimension, state_dimension, number_steps])
        P2[:, :, 0] = initial_moments[1]
        Phi2 = state_transition_tensors[1]

        if order > 2:
            P3 = np.zeros([state_dimension, state_dimension, state_dimension, number_steps])
            P3[:, :, :, 0] = initial_moments[2]
            Phi3 = state_transition_tensors[2]

            if order > 3:
                P4 = np.zeros([state_dimension, state_dimension, state_dimension, state_dimension, number_steps])
                P4[:, :, :, :, 0] = initial_moments[3]
                Phi4 = state_transition_tensors[3]


    ## ODEs for higher-order state transition tensors (linear ODEs)
    if order == 1:

        for k in range(1, number_steps):
            print(k)
            P1[:, k] = np.tensordot(Phi1[:, :, k], P1[:, 0], axes=([1], [0]))

        return [P1]


    if order == 2:

        for k in range(1, number_steps):
            print(k)
            P1[:, k] = np.tensordot(Phi1[:, :, k], P1[:, 0], axes=([1], [0])) + \
                       np.tensordot(Phi2[:, :, :, k], P2[:, :, 0], axes=([1, 2], [0, 1])) / 2
            P2[:, :, k] = np.tensordot(np.tensordot(Phi1[:, :, k], Phi1[:, :, k], axes=0), P2[:, :, 0], axes=([1, 3], [0, 1]))

        return [P1, P2]


    if order == 3:

        for k in range(1, number_steps):
            print(k)
            P1[:, k] = np.tensordot(Phi1[:, :, k], P1[:, 0], axes=([1], [0])) + \
                       np.tensordot(Phi2[:, :, :, k], P2[:, :, 0], axes=([1, 2], [0, 1])) / 2 + \
                       np.tensordot(Phi3[:, :, :, :, k], P3[:, :, :, 0], axes=([1, 2, 3], [0, 1, 2])) / 6
            P2[:, :, k] = np.tensordot(np.tensordot(Phi1[:, :, k], Phi1[:, :, k], axes=0), P2[:, :, 0], axes=([1, 3], [0, 1])) + \
                          np.tensordot(np.tensordot(Phi2[:, :, :, k], Phi1[:, :, k], axes=0), P3[:, :, :, 0], axes=([1, 2, 4], [0, 1, 2])) / 2 + \
                          np.tensordot(np.tensordot(Phi1[:, :, k], Phi2[:, :, :, k], axes=0), P3[:, :, :, 0], axes=([1, 3, 4], [0, 1, 2])) / 2
            P3[:, :, :, k] = np.tensordot(np.tensordot(np.tensordot(Phi1[:, :, k], Phi1[:, :, k], axes=0), Phi1[:, :, k], axes=0), P3[:, :, :, 0], axes=([1, 3, 5], [0, 1, 2]))

        return [P1, P2, P3]


    if order == 4:

        for k in range(1, number_steps):
            print(k)
            P1[:, k] = np.tensordot(Phi1[:, :, k], P1[:, 0], axes=([1], [0])) + \
                       np.tensordot(Phi2[:, :, :, k], P2[:, :, 0], axes=([1, 2], [0, 1])) / 2 + \
                       np.tensordot(Phi3[:, :, :, :, k], P3[:, :, :, 0], axes=([1, 2, 3], [0, 1, 2])) / 6 + \
                       np.tensordot(Phi4[:, :, :, :, :, k], P4[:, :, :, :, 0], axes=([1, 2, 3, 4], [0, 1, 2, 3])) / 24
            P2[:, :, k] = np.tensordot(np.tensordot(Phi1[:, :, k], Phi1[:, :, k], axes=0), P2[:, :, 0], axes=([1, 3], [0, 1])) + \
                          np.tensordot(np.tensordot(Phi2[:, :, :, k], Phi1[:, :, k], axes=0), P3[:, :, :, 0], axes=([1, 2, 4], [0, 1, 2])) / 2 + \
                          np.tensordot(np.tensordot(Phi1[:, :, k], Phi2[:, :, :, k], axes=0), P3[:, :, :, 0], axes=([1, 3, 4], [0, 1, 2])) / 2 + \
                          np.tensordot(np.tensordot(Phi1[:, :, k], Phi3[:, :, :, :, k], axes=0), P4[:, :, :, :, 0], axes=([1, 3, 4, 5], [0, 1, 2, 3])) / 6 + \
                          np.tensordot(np.tensordot(Phi3[:, :, :, :, k], Phi1[:, :, k], axes=0), P4[:, :, :, :, 0], axes=([1, 2, 3, 5], [0, 1, 2, 3])) / 6 + \
                          np.tensordot(np.tensordot(Phi2[:, :, :, k], Phi2[:, :, :, k], axes=0), P4[:, :, :, :, 0], axes=([1, 2, 4, 5], [0, 1, 2, 3])) / 4
            P3[:, :, :, k] = np.tensordot(np.tensordot(np.tensordot(Phi1[:, :, k], Phi1[:, :, k], axes=0), Phi1[:, :, k], axes=0), P3[:, :, :, 0], axes=([1, 3, 5], [0, 1, 2])) + \
                             np.tensordot(np.tensordot(np.tensordot(Phi2[:, :, :, k], Phi1[:, :, k], axes=0), Phi1[:, :, k], axes=0), P4[:, :, :, :, 0], axes=([1, 2, 4, 6], [0, 1, 2, 3])) / 2 + \
                             np.tensordot(np.tensordot(np.tensordot(Phi1[:, :, k], Phi2[:, :, :, k], axes=0), Phi1[:, :, k], axes=0), P4[:, :, :, :, 0], axes=([1, 3, 4, 6], [0, 1, 2, 3])) / 2 + \
                             np.tensordot(np.tensordot(np.tensordot(Phi1[:, :, k], Phi1[:, :, k], axes=0), Phi2[:, :, :, k], axes=0), P4[:, :, :, :, 0], axes=([1, 3, 5, 6], [0, 1, 2, 3])) / 2
            P4[:, :, :, :, k] = np.tensordot(np.tensordot(np.tensordot(np.tensordot(Phi1[:, :, k], Phi1[:, :, k], axes=0), Phi1[:, :, k], axes=0), Phi1[:, :, k], axes=0), P4[:, :, :, :, 0], axes=([1, 3, 5, 7], [0, 1, 2, 3]))

        return [P1, P2, P3, P4]
