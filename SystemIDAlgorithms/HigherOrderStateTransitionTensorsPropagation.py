"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 17
Date: October 2021
Python: 3.7.7
"""


import numpy as np
from scipy.integrate import odeint


def higherOrderStateTransitionTensorsPropagation(sensitivities, tk, dt):

    ## Order of the expansion and dimension of the system
    order = len(sensitivities)
    dimension, _ = sensitivities[0](0).shape


    ## Initialization
    Phi1 = np.eye(dimension)
    Ac1 = sensitivities[0]

    if order > 1:
        Phi2 = np.zeros([dimension, dimension, dimension])
        Ac2 = sensitivities[1]

        if order > 2:
            Phi3 = np.zeros([dimension, dimension, dimension, dimension])
            Ac3 = sensitivities[2]

            if order > 3:
                Phi4 = np.zeros([dimension, dimension, dimension, dimension, dimension])
                Ac4 = sensitivities[3]


    ## ODEs for higher-order state transition tensors (linear ODEs)
    if order == 1:

        def dPhi(Phi, t):
            Phi1_tensor = Phi.reshape(dimension, dimension)
            return np.tensordot(Ac1(t), Phi1_tensor, axes=([1], [0])).reshape(dimension ** 2)

        A1_vec = odeint(dPhi, Phi1.reshape(dimension**2), np.array([tk, tk + dt]), rtol=1e-13, atol=1e-13)
        A1_tensor = A1_vec[-1, :].reshape(dimension, dimension)

        return A1_tensor


    if order == 2:

        def dPhi(Phi, t):
            Phi1_tensor = Phi[0:dimension ** 2].reshape(dimension, dimension)
            Phi2_tensor = Phi[dimension ** 2:].reshape(dimension, dimension, dimension)
            dPhi1_tensor = np.tensordot(Ac1(t), Phi1_tensor, axes=([1], [0]))
            dPhi2_tensor = np.tensordot(Ac1(t), Phi2_tensor, axes=([1], [0])) + np.tensordot(Ac2(t), np.tensordot(Phi1_tensor, Phi1_tensor, axes=0), axes=([1, 2], [0, 2]))
            return np.concatenate((dPhi1_tensor.reshape(dimension ** 2), dPhi2_tensor.reshape(dimension ** 3)))

        A_vec = odeint(dPhi, np.concatenate((Phi1.reshape(dimension ** 2), Phi2.reshape(dimension ** 3))), np.array([tk, tk + dt]), rtol=1e-13, atol=1e-13)
        A1_tensor = A_vec[-1, 0:dimension ** 2].reshape(dimension, dimension)
        A2_tensor = A_vec[-1, dimension ** 2:].reshape(dimension, dimension, dimension)

        return A1_tensor, A2_tensor


    if order == 3:

        def dPhi(Phi, t):
            Phi1_tensor = Phi[0:dimension ** 2].reshape(dimension, dimension)
            Phi2_tensor = Phi[dimension ** 2:dimension ** 2 + dimension ** 3].reshape(dimension, dimension, dimension)
            Phi3_tensor = Phi[dimension ** 2 + dimension ** 3:].reshape(dimension, dimension, dimension, dimension)
            dPhi1_tensor = np.tensordot(Ac1(t), Phi1_tensor, axes=([1], [0]))
            dPhi2_tensor = np.tensordot(Ac1(t), Phi2_tensor, axes=([1], [0])) + \
                           np.tensordot(Ac2(t), np.tensordot(Phi1_tensor, Phi1_tensor, axes=0), axes=([1, 2], [0, 2]))
            dPhi3_tensor = np.tensordot(Ac1(t), Phi3_tensor, axes=([1], [0])) + \
                           np.tensordot(Ac2(t), np.tensordot(Phi2_tensor, Phi1_tensor, axes=0), axes=([1, 2], [0, 3])) + \
                           np.transpose(np.tensordot(Ac2(t), np.tensordot(Phi2_tensor, Phi1_tensor, axes=0), axes=([1, 2], [0, 3])), axes=[0, 2, 3, 1]) + \
                           np.transpose(np.tensordot(Ac2(t), np.tensordot(Phi2_tensor, Phi1_tensor, axes=0), axes=([1, 2], [0, 3])), axes=[0, 3, 1, 2]) + \
                           np.tensordot(Ac3(t), np.tensordot(Phi1_tensor, np.tensordot(Phi1_tensor, Phi1_tensor, axes=0), axes=0), axes=([1, 2, 3], [0, 2, 4]))

            return np.concatenate((dPhi1_tensor.reshape(dimension ** 2), dPhi2_tensor.reshape(dimension ** 3), dPhi3_tensor.reshape(dimension ** 4)))

        A_vec = odeint(dPhi, np.concatenate((Phi1.reshape(dimension ** 2), Phi2.reshape(dimension ** 3), Phi3.reshape(dimension ** 4))), np.array([tk, tk + dt]), rtol=1e-13, atol=1e-13)
        A1_tensor = A_vec[-1, 0:dimension ** 2].reshape(dimension, dimension)
        A2_tensor = A_vec[-1, dimension ** 2:dimension ** 2 + dimension ** 3].reshape(dimension, dimension, dimension)
        A3_tensor = A_vec[-1, dimension ** 2 + dimension ** 3:].reshape(dimension, dimension, dimension, dimension)

        return A1_tensor, A2_tensor, A3_tensor


    if order == 4:

        def dPhi(Phi, t):
            Phi1_tensor = Phi[0:dimension ** 2].reshape(dimension, dimension)
            Phi2_tensor = Phi[dimension ** 2:dimension ** 2 + dimension ** 3].reshape(dimension, dimension, dimension)
            Phi3_tensor = Phi[dimension ** 2 + dimension ** 3:dimension ** 2 + dimension ** 3 + dimension ** 4].reshape(dimension, dimension, dimension, dimension)
            Phi4_tensor = Phi[dimension ** 2 + dimension ** 3 + dimension ** 4:].reshape(dimension, dimension, dimension, dimension, dimension)

            dPhi1_tensor = np.tensordot(Ac1(t), Phi1_tensor, axes=([1], [0]))

            dPhi2_tensor = np.tensordot(Ac1(t), Phi2_tensor, axes=([1], [0])) + \
                           np.tensordot(Ac2(t), np.tensordot(Phi1_tensor, Phi1_tensor, axes=0), axes=([1, 2], [0, 2]))

            dPhi3_tensor = np.tensordot(Ac1(t), Phi3_tensor, axes=([1], [0])) + \
                           np.tensordot(Ac2(t), np.tensordot(Phi2_tensor, Phi1_tensor, axes=0), axes=([1, 2], [0, 3])) + \
                           np.transpose(np.tensordot(Ac2(t), np.tensordot(Phi2_tensor, Phi1_tensor, axes=0), axes=([1, 2], [0, 3])), axes=[0, 1, 3, 2]) + \
                           np.tensordot(Ac2(t), np.tensordot(Phi1_tensor, Phi2_tensor, axes=0), axes=([1, 2], [0, 2])) + \
                           np.tensordot(Ac3(t), np.tensordot(Phi1_tensor, np.tensordot(Phi1_tensor, Phi1_tensor, axes=0), axes=0), axes=([1, 2, 3], [0, 2, 4]))

            dPhi4_tensor = np.tensordot(Ac1(t), Phi4_tensor, axes=([1], [0])) + \
                           np.tensordot(Ac2(t), np.tensordot(Phi3_tensor, Phi1_tensor, axes=0), axes=([1, 2], [0, 4])) + \
                           np.transpose(np.tensordot(Ac2(t), np.tensordot(Phi3_tensor, Phi1_tensor, axes=0), axes=([1, 2], [0, 4])), axes=[0, 1, 2, 4, 3]) + \
                           np.transpose(np.tensordot(Ac2(t), np.tensordot(Phi3_tensor, Phi1_tensor, axes=0), axes=([1, 2], [0, 4])), axes=[0, 1, 4, 2, 3]) + \
                           np.tensordot(Ac2(t), np.tensordot(Phi2_tensor, Phi2_tensor, axes=0), axes=([1, 2], [0, 3])) + \
                           np.transpose(np.tensordot(Ac2(t), np.tensordot(Phi2_tensor, Phi2_tensor, axes=0), axes=([1, 2], [0, 3])), axes=[0, 1, 3, 2, 4]) + \
                           np.transpose(np.tensordot(Ac2(t), np.tensordot(Phi2_tensor, Phi2_tensor, axes=0), axes=([1, 2], [0, 3])), axes=[0, 1, 3, 4, 2]) + \
                           np.tensordot(Ac2(t), np.tensordot(Phi1_tensor, Phi3_tensor, axes=0), axes=([1, 2], [0, 2])) + \
                           np.tensordot(Ac3(t), np.tensordot(np.tensordot(Phi2_tensor, Phi1_tensor, axes=0), Phi1_tensor, axes=0), axes=([1, 2, 3], [0, 3, 5])) + \
                           np.transpose(np.tensordot(Ac3(t), np.tensordot(np.tensordot(Phi2_tensor, Phi1_tensor, axes=0), Phi1_tensor, axes=0), axes=([1, 2, 3], [0, 3, 5])), axes=[0, 1, 3, 2, 4]) + \
                           np.transpose(np.tensordot(Ac3(t), np.tensordot(np.tensordot(Phi2_tensor, Phi1_tensor, axes=0), Phi1_tensor, axes=0), axes=([1, 2, 3], [0, 3, 5])), axes=[0, 1, 3, 4, 2]) + \
                           np.tensordot(Ac3(t), np.tensordot(np.tensordot(Phi1_tensor, Phi2_tensor, axes=0), Phi1_tensor, axes=0), axes=([1, 2, 3], [0, 2, 5])) + \
                           np.transpose(np.tensordot(Ac3(t), np.tensordot(np.tensordot(Phi1_tensor, Phi2_tensor, axes=0), Phi1_tensor, axes=0), axes=([1, 2, 3], [0, 2, 5])), axes=[0, 1, 2, 4, 3]) + \
                           np.tensordot(Ac3(t), np.tensordot(np.tensordot(Phi1_tensor, Phi1_tensor, axes=0), Phi2_tensor, axes=0), axes=([1, 2, 3], [0, 2, 4])) + \
                           np.tensordot(Ac4(t), np.tensordot(np.tensordot(np.tensordot(Phi1_tensor, Phi1_tensor, axes=0), Phi1_tensor, axes=0), Phi1_tensor, axes=0), axes=([1, 2, 3, 4], [0, 2, 4, 6]))

            return np.concatenate((dPhi1_tensor.reshape(dimension ** 2), dPhi2_tensor.reshape(dimension ** 3), dPhi3_tensor.reshape(dimension ** 4), dPhi4_tensor.reshape(dimension ** 5)))

        Phi0 = np.concatenate((Phi1.reshape(dimension ** 2), Phi2.reshape(dimension ** 3), Phi3.reshape(dimension ** 4), Phi4.reshape(dimension ** 5)))
        A_vec = odeint(dPhi, Phi0, np.array([tk, tk + dt]), rtol=1e-13, atol=1e-13)
        A1_tensor = A_vec[-1, 0:dimension ** 2].reshape(dimension, dimension)
        A2_tensor = A_vec[-1, dimension ** 2:dimension ** 2 + dimension ** 3].reshape(dimension, dimension, dimension)
        A3_tensor = A_vec[-1, dimension ** 2 + dimension ** 3:dimension ** 2 + dimension ** 3 + dimension ** 4].reshape(dimension, dimension, dimension, dimension)
        A4_tensor = A_vec[-1, dimension ** 2 + dimension ** 3 + dimension ** 4:].reshape(dimension, dimension, dimension, dimension, dimension)

    return A1_tensor, A2_tensor, A3_tensor, A4_tensor































# def dPhi(Phi, t):
#     Phi1_tensor = Phi[0:dimension ** 2].reshape(dimension, dimension)
#     Phi2_tensor = Phi[dimension ** 2:dimension ** 2 + dimension ** 3].reshape(dimension, dimension, dimension)
#     Phi3_tensor = Phi[dimension ** 2 + dimension ** 3:dimension ** 2 + dimension ** 3 + dimension ** 4].reshape(
#         dimension, dimension, dimension, dimension)
#     Phi4_tensor = Phi[dimension ** 2 + dimension ** 3 + dimension ** 4:].reshape(dimension, dimension, dimension,
#                                                                                  dimension, dimension)
#     dPhi1_tensor = np.zeros([dimension, dimension])
#     dPhi2_tensor = np.zeros([dimension, dimension, dimension])
#     dPhi3_tensor = np.zeros([dimension, dimension, dimension, dimension])
#     dPhi4_tensor = np.zeros([dimension, dimension, dimension, dimension, dimension])
#     for i in range(dimension):
#         for j1 in range(dimension):
#             for r1 in range(dimension):
#                 dPhi1_tensor[i, j1] += Ac1(t)[i, r1] * Phi1_tensor[r1, j1]
#     for i in range(dimension):
#         for j1 in range(dimension):
#             for j2 in range(dimension):
#                 for r1 in range(dimension):
#                     dPhi2_tensor[i, j1, j2] += Ac1(t)[i, r1] * Phi2_tensor[r1, j1, j2]
#                     for r2 in range(dimension):
#                         dPhi2_tensor[i, j1, j2] += Ac2(t)[i, r1, r2] * Phi1_tensor[r1, j1] * Phi1_tensor[r2, j2]
#     for i in range(dimension):
#         for j1 in range(dimension):
#             for j2 in range(dimension):
#                 for j3 in range(dimension):
#                     for r1 in range(dimension):
#                         dPhi3_tensor[i, j1, j2, j3] += Ac1(t)[i, r1] * Phi3_tensor[r1, j1, j2, j3]
#                         for r2 in range(dimension):
#                             dPhi3_tensor[i, j1, j2, j3] += Ac2(t)[i, r1, r2] * (
#                                         Phi2_tensor[r1, j1, j2] * Phi1_tensor[r2, j3] + Phi2_tensor[r2, j2, j3] *
#                                         Phi1_tensor[r1, j1] + Phi2_tensor[r1, j1, j3] * Phi1_tensor[r2, j2])
#                             for r3 in range(dimension):
#                                 dPhi3_tensor[i, j1, j2, j3] += Ac3(t)[i, r1, r2, r3] * Phi1_tensor[r1, j1] * \
#                                                                Phi1_tensor[r2, j2] * Phi1_tensor[r3, j3]
#     for i in range(dimension):
#         for j1 in range(dimension):
#             for j2 in range(dimension):
#                 for j3 in range(dimension):
#                     for j4 in range(dimension):
#                         for r1 in range(dimension):
#                             dPhi4_tensor[i, j1, j2, j3, j4] += Ac1(t)[i, r1] * Phi4_tensor[r1, j1, j2, j3, j4]
#                             for r2 in range(dimension):
#                                 dPhi4_tensor[i, j1, j2, j3, j4] += Ac2(t)[i, r1, r2] * Phi3_tensor[r1, j1, j2, j3] * \
#                                                                    Phi1_tensor[r2, j4] + \
#                                                                    Ac2(t)[i, r1, r2] * Phi3_tensor[r1, j1, j2, j4] * \
#                                                                    Phi1_tensor[r2, j3] + \
#                                                                    Ac2(t)[i, r1, r2] * Phi3_tensor[r1, j1, j3, j4] * \
#                                                                    Phi1_tensor[r2, j2] + \
#                                                                    Ac2(t)[i, r1, r2] * Phi2_tensor[r1, j1, j2] * \
#                                                                    Phi2_tensor[r2, j3, j4] + \
#                                                                    Ac2(t)[i, r1, r2] * Phi2_tensor[r1, j1, j3] * \
#                                                                    Phi2_tensor[r2, j2, j4] + \
#                                                                    Ac2(t)[i, r1, r2] * Phi2_tensor[r1, j1, j4] * \
#                                                                    Phi2_tensor[r2, j2, j3] + \
#                                                                    Ac2(t)[i, r1, r2] * Phi1_tensor[r1, j1] * \
#                                                                    Phi3_tensor[r2, j2, j3, j4]
#                                 for r3 in range(dimension):
#                                     dPhi4_tensor[i, j1, j2, j3, j4] += Ac3(t)[i, r1, r2, r3] * Phi2_tensor[r1, j1, j2] * \
#                                                                        Phi1_tensor[r2, j3] * Phi1_tensor[r3, j4] + \
#                                                                        Ac3(t)[i, r1, r2, r3] * Phi2_tensor[r1, j1, j3] * \
#                                                                        Phi1_tensor[r2, j2] * Phi1_tensor[r3, j4] + \
#                                                                        Ac3(t)[i, r1, r2, r3] * Phi2_tensor[r1, j1, j4] * \
#                                                                        Phi1_tensor[r2, j2] * Phi1_tensor[r3, j3] + \
#                                                                        Ac3(t)[i, r1, r2, r3] * Phi1_tensor[r1, j1] * \
#                                                                        Phi2_tensor[r2, j2, j3] * Phi1_tensor[r3, j4] + \
#                                                                        Ac3(t)[i, r1, r2, r3] * Phi1_tensor[r1, j1] * \
#                                                                        Phi2_tensor[r2, j2, j4] * Phi1_tensor[r3, j3] + \
#                                                                        Ac3(t)[i, r1, r2, r3] * Phi1_tensor[r1, j1] * \
#                                                                        Phi1_tensor[r2, j2] * Phi2_tensor[r3, j3, j4]
#                                     for r4 in range(dimension):
#                                         dPhi4_tensor[i, j1, j2, j3, j4] += Ac4(t)[i, r1, r2, r3, r4] * Phi1_tensor[
#                                             r1, j1] * Phi1_tensor[r2, j2] * Phi1_tensor[r3, j3] * Phi1_tensor[r4, j4]
#
#     return np.concatenate((dPhi1_tensor.reshape(dimension ** 2), dPhi2_tensor.reshape(dimension ** 3),
#                            dPhi3_tensor.reshape(dimension ** 4), dPhi4_tensor.reshape(dimension ** 5)))




