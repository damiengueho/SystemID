"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


import numpy as np
from scipy.integrate import odeint

from systemID.SystemIDAlgorithms.Integrate import integrate


def propagation(signal, system, **kwargs):

    # Get system's parameters
    state_dimension = system.state_dimension
    output_dimension = system.output_dimension
    x0 = system.x0
    initial_states = system.initial_states
    number_initial_states = len(initial_states)

    # Propagate depending on system type
    system_type = system.system_type

    if system_type == 'Discrete Linear':
        dt = system.dt
        number_steps = signal.number_steps
        (A, B, C, D) = system.A, system.B, system.C, system.D

        u = signal.data

        observer = kwargs.get('observer', False)
        if observer:
            K = system.K
            reference_output_signal = kwargs.get('reference_output_signal', np.zeros([output_dimension, number_steps]))

        process_noise = kwargs.get('process_noise', False)
        process_noise_signal = kwargs.get('process_noise_signal')
        measurement_noise = kwargs.get('measurement_noise', False)
        measurement_noise_signal = kwargs.get('measurement_noise_signal')

        x = np.zeros([state_dimension, number_steps + 1])
        x[:, 0] = x0
        y = np.zeros([output_dimension, number_steps])
        count_init_states = 1

        for i in range(number_steps):
            if measurement_noise:
                y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i * dt), u[:, i]) + measurement_noise_signal.data[:, i]
            else:
                y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i * dt), u[:, i])
            if number_initial_states > 1:
                if i + 1 == initial_states[count_init_states][1]:
                    x[:, i + 1] = initial_states[count_init_states][0]
                    if count_init_states < number_initial_states - 1:
                        count_init_states += 1
                else:
                    if observer:
                        if process_noise:
                            x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i])) + process_noise_signal.data[:, i]
                        else:
                            x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i]))
                    else:
                        if process_noise:
                            x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i]) + process_noise_signal.data[:, i]
                        else:
                            x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i])
            else:
                if observer:
                    if process_noise:
                        x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i])) + process_noise_signal.data[:, i]
                    else:
                        x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i]))
                else:
                    if process_noise:
                        x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i]) + process_noise_signal.data[:, i]
                    else:
                        x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i])


    # if system_type == 'Continuous Order 2':
    #     dt = system.dt
    #     number_steps = signal.number_steps
    #     Acs, F, G = system.Acs, system.F, system.G
    #
    #     u = signal.u
    #
    #     observer = kwargs.get('observer', False)
    #     if observer:
    #         K = system.K
    #         reference_output_signal = kwargs.get('reference_output_signal', np.zeros([output_dimension, number_steps]))
    #
    #     process_noise = kwargs.get('process_noise', False)
    #     process_noise_signal = kwargs.get('process_noise_signal')
    #     measurement_noise = kwargs.get('measurement_noise', False)
    #     measurement_noise_signal = kwargs.get('measurement_noise_signal')
    #
    #     x = np.zeros([state_dimension, number_steps + 1])
    #     x[:, 0] = x0
    #     y = np.zeros([output_dimension, number_steps])
    #     count_init_states = 1
    #
    #     higherOrderStateTransitionTensorsPropagation(Acs, F, u, x0, tspan)
    #
    #     for i in range(number_steps):
    #         if measurement_noise:
    #             y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i * dt), u[:, i]) + measurement_noise_signal.data[:, i]
    #         else:
    #             y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i*dt), u[:, i])
    #         if number_initial_states > 1:
    #             if i + 1 == initial_states[count_init_states][1]:
    #                 x[:, i + 1] = initial_states[count_init_states][0]
    #                 if count_init_states < number_initial_states - 1:
    #                     count_init_states += 1
    #             else:
    #                 if observer:
    #                     if process_noise:
    #                         x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + np.matmul(B(i * dt), u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i])) + process_noise_signal.data[:, i]
    #                     else:
    #                         x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + np.matmul(B(i * dt), u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i]))
    #                 else:
    #                     if process_noise:
    #                         x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + np.matmul(B(i * dt), u[:, i]) + process_noise_signal.data[:, i]
    #                     else:
    #                         x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + np.matmul(B(i * dt), u[:, i])
    #         else:
    #             if observer:
    #                 if process_noise:
    #                     x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + np.matmul(B(i * dt), u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i])) + process_noise_signal.data[:, i]
    #                 else:
    #                     x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + np.matmul(B(i * dt), u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i]))
    #             else:
    #                 if process_noise:
    #                     x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + np.matmul(B(i * dt), u[:, i]) + process_noise_signal.data[:, i]
    #                 else:
    #                     x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.tensordot(x[:, i], x[:, i], axes=0), axes=([1, 2], [0, 1])) + np.matmul(B(i * dt), u[:, i])
    #
    #
    # if system_type == 'Discrete Linear Order 3':
    #     dt = system.dt
    #     number_steps = signal.number_steps
    #     A3, B, C, D = system.A3, system.B, system.C, system.D
    #
    #     u = signal.data
    #
    #     observer = kwargs.get('observer', False)
    #     if observer:
    #         K = system.K
    #         reference_output_signal = kwargs.get('reference_output_signal', np.zeros([output_dimension, number_steps]))
    #
    #     process_noise = kwargs.get('process_noise', False)
    #     process_noise_signal = kwargs.get('process_noise_signal')
    #     measurement_noise = kwargs.get('measurement_noise', False)
    #     measurement_noise_signal = kwargs.get('measurement_noise_signal')
    #
    #     x = np.zeros([state_dimension, number_steps + 1])
    #     x[:, 0] = x0
    #     y = np.zeros([output_dimension, number_steps])
    #     count_init_states = 1
    #
    #     for i in range(number_steps):
    #         At1, At2, At3 = A3(i * dt)
    #         if measurement_noise:
    #             y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i * dt), u[:, i]) + measurement_noise_signal.data[:, i]
    #         else:
    #             y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i*dt), u[:, i])
    #         if number_initial_states > 1:
    #             if i + 1 == initial_states[count_init_states][1]:
    #                 x[:, i + 1] = initial_states[count_init_states][0]
    #                 if count_init_states < number_initial_states - 1:
    #                     count_init_states += 1
    #             else:
    #                 if observer:
    #                     if process_noise:
    #                         x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(At3, np.tensordot(np.outer(x[:, i], x[:, i]), x[:, i], axes=0), axes=([1, 2, 3], [0, 1, 2])) + np.matmul(B(i * dt), u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i])) + process_noise_signal.data[:, i]
    #                     else:
    #                         x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(At3, np.tensordot(np.outer(x[:, i], x[:, i]), x[:, i], axes=0), axes=([1, 2, 3], [0, 1, 2])) + np.matmul(B(i * dt), u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i]))
    #                 else:
    #                     if process_noise:
    #                         x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(At3, np.tensordot(np.outer(x[:, i], x[:, i]), x[:, i], axes=0), axes=([1, 2, 3], [0, 1, 2])) + np.matmul(B(i * dt), u[:, i]) + process_noise_signal.data[:, i]
    #                     else:
    #                         x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(At3, np.tensordot(np.outer(x[:, i], x[:, i]), x[:, i], axes=0), axes=([1, 2, 3], [0, 1, 2])) + np.matmul(B(i * dt), u[:, i])
    #         else:
    #             if observer:
    #                 if process_noise:
    #                     x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(At3, np.tensordot(np.outer(x[:, i], x[:, i]), x[:, i], axes=0), axes=([1, 2, 3], [0, 1, 2])) + np.matmul(B(i * dt), u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i])) + process_noise_signal.data[:, i]
    #                 else:
    #                     x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(At3, np.tensordot(np.outer(x[:, i], x[:, i]), x[:, i], axes=0), axes=([1, 2, 3], [0, 1, 2])) + np.matmul(B(i * dt), u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i]))
    #             else:
    #                 if process_noise:
    #                     x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(At3, np.tensordot(np.outer(x[:, i], x[:, i]), x[:, i], axes=0), axes=([1, 2, 3], [0, 1, 2])) + np.matmul(B(i * dt), u[:, i]) + process_noise_signal.data[:, i]
    #                 else:
    #                     x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.tensordot(x[:, i], x[:, i], axes=0), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(At3, np.tensordot(np.tensordot(x[:, i], x[:, i], axes=0), x[:, i], axes=0), axes=([1, 2, 3], [0, 1, 2])) + np.matmul(B(i * dt), u[:, i])
    #
    #
    # if system_type == 'Discrete Linear Order 4':
    #     dt = system.dt
    #     number_steps = signal.number_steps
    #     A4, B, C, D = system.A4, system.B, system.C, system.D
    #
    #     u = signal.data
    #
    #     observer = kwargs.get('observer', False)
    #     if observer:
    #         K = system.K
    #         reference_output_signal = kwargs.get('reference_output_signal', np.zeros([output_dimension, number_steps]))
    #
    #     process_noise = kwargs.get('process_noise', False)
    #     process_noise_signal = kwargs.get('process_noise_signal')
    #     measurement_noise = kwargs.get('measurement_noise', False)
    #     measurement_noise_signal = kwargs.get('measurement_noise_signal')
    #
    #     x = np.zeros([state_dimension, number_steps + 1])
    #     x[:, 0] = x0
    #     y = np.zeros([output_dimension, number_steps])
    #     count_init_states = 1
    #
    #     for i in range(number_steps):
    #         At1, At2, At3, At4 = A4(i * dt)
    #         if measurement_noise:
    #             y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i * dt), u[:, i]) + measurement_noise_signal.data[:, i]
    #         else:
    #             y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i*dt), u[:, i])
    #         if number_initial_states > 1:
    #             if i + 1 == initial_states[count_init_states][1]:
    #                 x[:, i + 1] = initial_states[count_init_states][0]
    #                 if count_init_states < number_initial_states - 1:
    #                     count_init_states += 1
    #             else:
    #                 if observer:
    #                     if process_noise:
    #                         x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(At3, np.tensordot(np.outer(x[:, i], x[:, i]), x[:, i], axes=0), axes=([1, 2, 3], [0, 1, 2])) + (1 / 24) * np.tensordot(At4, np.tensordot(np.tensordot(np.tensordot(x[:, i], x[:, i], axes=0), x[:, i], axes=0), x[:, i], axes=0), axes=([1, 2, 3, 4], [0, 1, 2, 3])) + np.matmul(B(i * dt), u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i])) + process_noise_signal.data[:, i]
    #                     else:
    #                         x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(At3, np.tensordot(np.outer(x[:, i], x[:, i]), x[:, i], axes=0), axes=([1, 2, 3], [0, 1, 2])) + (1 / 24) * np.tensordot(At4, np.tensordot(np.tensordot(np.tensordot(x[:, i], x[:, i], axes=0), x[:, i], axes=0), x[:, i], axes=0), axes=([1, 2, 3, 4], [0, 1, 2, 3])) + np.matmul(B(i * dt), u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i]))
    #                 else:
    #                     if process_noise:
    #                         x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(At3, np.tensordot(np.outer(x[:, i], x[:, i]), x[:, i], axes=0), axes=([1, 2, 3], [0, 1, 2])) + (1 / 24) * np.tensordot(At4, np.tensordot(np.tensordot(np.tensordot(x[:, i], x[:, i], axes=0), x[:, i], axes=0), x[:, i], axes=0), axes=([1, 2, 3, 4], [0, 1, 2, 3])) + np.matmul(B(i * dt), u[:, i]) + process_noise_signal.data[:, i]
    #                     else:
    #                         x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(At3, np.tensordot(np.outer(x[:, i], x[:, i]), x[:, i], axes=0), axes=([1, 2, 3], [0, 1, 2])) + (1 / 24) * np.tensordot(At4, np.tensordot(np.tensordot(np.tensordot(x[:, i], x[:, i], axes=0), x[:, i], axes=0), x[:, i], axes=0), axes=([1, 2, 3, 4], [0, 1, 2, 3])) + np.matmul(B(i * dt), u[:, i])
    #         else:
    #             if observer:
    #                 if process_noise:
    #                     x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(At3, np.tensordot(np.outer(x[:, i], x[:, i]), x[:, i], axes=0), axes=([1, 2, 3], [0, 1, 2])) + (1 / 24) * np.tensordot(At4, np.tensordot(np.tensordot(np.tensordot(x[:, i], x[:, i], axes=0), x[:, i], axes=0), x[:, i], axes=0), axes=([1, 2, 3, 4], [0, 1, 2, 3])) + np.matmul(B(i * dt), u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i])) + process_noise_signal.data[:, i]
    #                 else:
    #                     x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(At3, np.tensordot(np.outer(x[:, i], x[:, i]), x[:, i], axes=0), axes=([1, 2, 3], [0, 1, 2])) + (1 / 24) * np.tensordot(At4, np.tensordot(np.tensordot(np.tensordot(x[:, i], x[:, i], axes=0), x[:, i], axes=0), x[:, i], axes=0), axes=([1, 2, 3, 4], [0, 1, 2, 3])) + np.matmul(B(i * dt), u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i]))
    #             else:
    #                 if process_noise:
    #                     x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.outer(x[:, i], x[:, i]), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(At3, np.tensordot(np.outer(x[:, i], x[:, i]), x[:, i], axes=0), axes=([1, 2, 3], [0, 1, 2])) + (1 / 24) * np.tensordot(At4, np.tensordot(np.tensordot(np.tensordot(x[:, i], x[:, i], axes=0), x[:, i], axes=0), x[:, i], axes=0), axes=([1, 2, 3, 4], [0, 1, 2, 3])) + np.matmul(B(i * dt), u[:, i]) + process_noise_signal.data[:, i]
    #                 else:
    #                     x[:, i + 1] = np.matmul(At1, x[:, i]) + (1 / 2) * np.tensordot(At2, np.tensordot(x[:, i], x[:, i], axes=0), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(At3, np.tensordot(np.tensordot(x[:, i], x[:, i], axes=0), x[:, i], axes=0), axes=([1, 2, 3], [0, 1, 2])) + (1 / 24) * np.tensordot(At4, np.tensordot(np.tensordot(np.tensordot(x[:, i], x[:, i], axes=0), x[:, i], axes=0), x[:, i], axes=0), axes=([1, 2, 3, 4], [0, 1, 2, 3])) + np.matmul(B(i * dt), u[:, i])
    #


    if system_type == 'Discrete Nonlinear':
        dt = system.dt
        number_steps = signal.number_steps
        (F, G) = system.F, system.G

        u = signal.data

        observer = kwargs.get('observer', False)
        if observer:
            K = system.K
            reference_output_signal = kwargs.get('reference_output_signal', np.zeros([output_dimension, number_steps]))

        process_noise = kwargs.get('process_noise', False)
        process_noise_signal = kwargs.get('process_noise_signal')
        measurement_noise = kwargs.get('measurement_noise', False)
        measurement_noise_signal = kwargs.get('measurement_noise_signal')

        x = np.zeros([state_dimension, number_steps + 1])
        x[:, 0] = x0
        y = np.zeros([output_dimension, number_steps])
        count_init_states = 1

        for i in range(number_steps):
            if measurement_noise:
                y[:, i] = G(x[:, i], i * dt, u[:, i]) + measurement_noise_signal.u(i * dt)
            else:
                y[:, i] = G(x[:, i], i * dt, u[:, i])
            if number_initial_states > 1:
                if i + 1 == initial_states[count_init_states][1]:
                    x[:, i + 1] = initial_states[count_init_states][0]
                    if count_init_states < number_initial_states - 1:
                        count_init_states += 1
                else:
                    if observer:
                        if process_noise:
                            x[:, i + 1] = F(x[:, i], i * dt, u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i])) + process_noise_signal(i * dt)
                        else:
                            x[:, i + 1] = F(x[:, i], i * dt, u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i]))
                    else:
                        if process_noise:
                            x[:, i + 1] = F(x[:, i], i * dt, u[:, i]) + process_noise_signal(i * dt)
                        else:
                            x[:, i + 1] = F(x[:, i], i * dt, u[:, i])
            else:
                if observer:
                    if process_noise:
                        x[:, i + 1] = F(x[:, i], i * dt, u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i])) + process_noise_signal(i * dt)
                    else:
                        x[:, i + 1] = F(x[:, i], i * dt, u[:, i]) + np.matmul(K(i * dt), (y[:, i] - reference_output_signal.data[:, i]))
                else:
                    if process_noise:
                        x[:, i + 1] = F(x[:, i], i * dt, u[:, i]) + process_noise_signal(i * dt)
                    else:
                        x[:, i + 1] = F(x[:, i], i * dt, u[:, i])


    if system_type == 'Continuous Linear':
        tspan = kwargs.get('tspan', np.zeros(1))
        (A, B, C, D) = system.A, system.B, system.C, system.D

        def dynamics(x, t, u):
            return np.matmul(A(t), x) + np.matmul(B(t), u(t))

        fixed_step_size = kwargs.get('fixed_step_size', False)
        if fixed_step_size:
            integration_step = kwargs.get('integration_step', 0.001)
            sol = integrate(dynamics, x0, tspan, integration_step, args=(signal.u, ))
        else:
            sol = odeint(dynamics, x0, tspan, args=(signal.u, ), rtol=1e-13, atol=1e-13)

        x = sol.T
        y = np.zeros([output_dimension, tspan.shape[0]])
        i = 0
        for t in tspan:
            y[:, i] = np.matmul(C(t), x[:, i]) + np.matmul(D(t), signal.u(t))
            i += 1


    if system_type == 'Continuous Bilinear':
        tspan = kwargs.get('tspan', np.zeros(1))
        (A, N, B, C, D) = system.A, system.N, system.B, system.C, system.D

        def dynamics(x, t, u):
            return np.matmul(A(t), x) + np.matmul(N(t), np.kron(u(t), x)) + np.matmul(B(t), u(t))

        fixed_step_size = kwargs.get('fixed_step_size', False)
        if fixed_step_size:
            integration_step = kwargs.get('integration_step', 0.001)
            sol = integrate(dynamics, x0, tspan, integration_step, args=(signal.u, ))
        else:
            sol = odeint(dynamics, x0, tspan, args=(signal.u, ), rtol=1e-13, atol=1e-13)

        x = sol.T
        y = np.zeros([output_dimension, tspan.shape[0]])
        i = 0
        for t in tspan:
            y[:, i] = np.matmul(C(t), x[:, i]) + np.matmul(D(t), signal.u(t))
            i += 1


    if system_type == 'Continuous Nonlinear':
        tspan = kwargs.get('tspan', np.zeros(1))
        (F, G) = system.F, system.G

        fixed_step_size = kwargs.get('fixed_ste_size', False)
        if fixed_step_size:
            integration_step = kwargs.get('integration_step', 0.001)
            sol = integrate(F, x0, tspan, integration_step, args=(signal.u, ))
        else:
            sol = odeint(F, x0, tspan, args=(signal.u,), rtol=1e-13, atol=1e-13)

        x = sol.T
        y = np.zeros([output_dimension, tspan.shape[0]])
        i = 0
        for t in tspan:
            y[:, i] = G(x[:, i], t, signal.u(t))
            i += 1


    if system_type == 'Continuous Nonlinear Higher Order Expansion':
        tspan = kwargs.get('tspan', np.zeros(1))
        number_steps = len(tspan)

        G = system.G

        x = np.zeros([state_dimension, number_steps])
        x[:, 0] = x0
        y = np.zeros([output_dimension, number_steps])

        if system.order == 1:
            for i in range(number_steps):
                A1 = system.A_vec[i, state_dimension:state_dimension + state_dimension ** 2].reshape(state_dimension, state_dimension)
                x[:, i] = np.matmul(A1, x0)
                y[:, i] = G(x[:, i], tspan[i], signal.u(tspan[i]))

        if system.order == 2:
            for i in range(number_steps):
                A1 = system.A_vec[i, state_dimension:state_dimension + state_dimension ** 2].reshape(state_dimension, state_dimension)
                A2 = system.A_vec[i, state_dimension + state_dimension ** 2:].reshape(state_dimension, state_dimension, state_dimension)
                x[:, i] = np.matmul(A1, x0) + (1 / 2) * np.tensordot(A2, np.tensordot(x0, x0, axes=0), axes=([1, 2], [0, 1]))
                y[:, i] = G(x[:, i], tspan[i], signal.u(tspan[i]))

        if system.order == 3:
            for i in range(number_steps):
                A1 = system.A_vec[i, state_dimension:state_dimension + state_dimension ** 2].reshape(state_dimension, state_dimension)
                A2 = system.A_vec[i, state_dimension + state_dimension ** 2:state_dimension + state_dimension ** 2 + state_dimension ** 3].reshape(state_dimension, state_dimension, state_dimension)
                A3 = system.A_vec[i, state_dimension + state_dimension ** 2 + state_dimension ** 3:].reshape(state_dimension, state_dimension, state_dimension, state_dimension)
                x[:, i] = np.matmul(A1, x0) + (1 / 2) * np.tensordot(A2, np.tensordot(x0, x0, axes=0), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(A3, np.tensordot(np.tensordot(x0, x0, axes=0), x0, axes=0), axes=([1, 2, 3], [0, 1, 2]))
                y[:, i] = G(x[:, i], tspan[i], signal.u(tspan[i]))

        if system.order == 4:
            for i in range(number_steps):
                A1 = system.A_vec[i, state_dimension:state_dimension + state_dimension ** 2].reshape(state_dimension, state_dimension)
                A2 = system.A_vec[i, state_dimension + state_dimension ** 2:state_dimension + state_dimension ** 2 + state_dimension ** 3].reshape(state_dimension, state_dimension, state_dimension)
                A3 = system.A_vec[i, state_dimension + state_dimension ** 2 + state_dimension ** 3:state_dimension + state_dimension ** 2 + state_dimension ** 3 + state_dimension ** 4].reshape(state_dimension, state_dimension, state_dimension, state_dimension)
                A4 = system.A_vec[i, state_dimension + state_dimension ** 2 + state_dimension ** 3 + state_dimension ** 4:].reshape(state_dimension, state_dimension, state_dimension, state_dimension, state_dimension)
                x[:, i] = np.matmul(A1, x0) + (1 / 2) * np.tensordot(A2, np.tensordot(x0, x0, axes=0), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(A3, np.tensordot(np.tensordot(x0, x0, axes=0), x0, axes=0), axes=([1, 2, 3], [0, 1, 2])) + (1 / 24) * np.tensordot(A4, np.tensordot(np.tensordot(np.tensordot(x0, x0, axes=0), x0, axes=0), x0, axes=0), axes=([1, 2, 3, 4], [0, 1, 2, 3]))
                y[:, i] = G(x[:, i], tspan[i], signal.u(tspan[i]))


    return (y, x)
