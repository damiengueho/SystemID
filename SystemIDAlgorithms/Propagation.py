"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 10
Date: April 2021
Python: 3.7.7
"""


import numpy as np
import scipy.linalg as LA
from scipy.integrate import odeint

from SystemIDAlgorithms.Integrate import integrate


def propagation(signal, system, **kwargs):

    # Get general parameters of the system
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

        x = np.zeros([state_dimension, number_steps + 1])
        x[:, 0] = x0
        y = np.zeros([output_dimension, number_steps])
        count_init_states = 1

        state_propagation = kwargs.get('state_propagation', False)
        signal_input_history = kwargs.get('signal_input_history', False)
        signal_output_history = kwargs.get('signal_output_history', False)

        # if state_propagation:
        #     A_list = []
        #     B_list = []
        #     C_list = []
        #     D_list = []
        #     nominal_input_history = []
        #     nominal_output_history = []
        #     for i in range(number_steps):
        #         A_list.append(A(i * dt))
        #         B_list.append(B(i * dt))
        #         C_list.append(C(i * dt))
        #         D_list.append(D(i * dt))
        #         nominal_input_history.append(signal_input_history[:, i])
        #         nominal_output_history.append(signal_output_history[:, i])
        #     nearest_index = 0
        #     for i in range(number_steps):
        #         C_mat = C_list[nearest_index]
        #         D_mat = D_list[nearest_index]
        #         y[:, i] = np.matmul(C_mat, x[:, i]) + np.matmul(D_mat, u[:, i])
        #         nearest_index = min(range(len(nominal_output_history)), key=lambda j: LA.norm(nominal_output_history[j] - (nominal_output_history[i] + y[:, i])) + LA.norm(nominal_input_history[j] - (nominal_input_history[i] + u[:, i])))
        #         if i % 5 == 0:
        #             print('i =', i)
        #             print('nominal_output_history[i] + y[:, i]', nominal_output_history[i] + y[:, i])
        #             print('nominal_output_history[nearest_index]', nominal_output_history[nearest_index])
        #             print('LA.norm(nominal_output_history[nearest_index] - (nominal_output_history[i] + y[:, i]))', LA.norm(nominal_output_history[nearest_index] - (nominal_output_history[i] + y[:, i])))
        #             print('---')
        #             print('nominal_input_history[i] + u[:, i]', nominal_input_history[i] + u[:, i])
        #             print('nominal_input_history[nearest_index]', nominal_input_history[nearest_index])
        #             print('LA.norm(nominal_input_history[nearest_index] - (nominal_input_history[i] + u[:, i]))', LA.norm(nominal_input_history[nearest_index] - (nominal_input_history[i] + u[:, i])))
        #             print('---')
        #             print('nearest_index = ', nearest_index)
        #             print('****************************************************************************************************************************')
        #         A_mat = A_list[nearest_index]
        #         B_mat = B_list[nearest_index]
        #         if number_initial_states > 1:
        #             if i + 1 == initial_states[count_init_states][1]:
        #                 x[:, i + 1] = initial_states[count_init_states][0]
        #                 if count_init_states < number_initial_states - 1:
        #                     count_init_states += 1
        #             else:
        #                 x[:, i + 1] = np.matmul(A_mat, x[:, i]) + np.matmul(B_mat, u[:, i])
        #         else:
        #             x[:, i + 1] = np.matmul(A_mat, x[:, i]) + np.matmul(B_mat, u[:, i])
        #
        if state_propagation:
            A_list = []
            B_list = []
            C_list = []
            D_list = []
            input_history = []
            output_history = []
            for i in range(number_steps):
                A_list.append(A(i * dt))
                B_list.append(B(i * dt))
                C_list.append(C(i * dt))
                D_list.append(D(i * dt))
                input_history.append(signal_input_history[:, i])
                output_history.append(signal_output_history[:, i])
            nearest_index_y = 0
            for i in range(number_steps):
                if i > 0:
                    nearest_index = min(range(len(output_history)), key=lambda j: (LA.norm(output_history[j] - y[:, i])))
                C_mat = C_list[nearest_index]
                D_mat = D_list[nearest_index]
                y[:, i] = np.matmul(C_mat, x[:, i]) + np.matmul(D_mat, u[:, i])
                #nearest_index_x = min(range(len(output_history)), key=lambda j: LA.norm(input_history[j] - u[:, i]))
                # if i % 5 == 0:
                #     print('i =', i)
                #     print('output_history[j]', output_history[nearest_index])
                #     print('y[:, i]', y[:, i])
                #     print('LA.norm(output_history[nearest_index] - y[:, i])', LA.norm(output_history[nearest_index] - y[:, i]))
                #     print('input_history[j]', input_history[nearest_index])
                #     print('u[:, i]', u[:, i])
                #     print('LA.norm(input_history[nearest_index] - u[:, i])', LA.norm(input_history[nearest_index] - u[:, i]))
                #     print('nearest_index = ', nearest_index)
                #     print('----------------------------------------------')
                A_mat = A_list[nearest_index]
                B_mat = B_list[nearest_index]
                if number_initial_states > 1:
                    if i + 1 == initial_states[count_init_states][1]:
                        x[:, i + 1] = initial_states[count_init_states][0]
                        if count_init_states < number_initial_states - 1:
                            count_init_states += 1
                    else:
                        x[:, i + 1] = np.matmul(A_mat, x[:, i]) + np.matmul(B_mat, u[:, i])
                else:
                    x[:, i + 1] = np.matmul(A_mat, x[:, i]) + np.matmul(B_mat, u[:, i])

        else:
            for i in range(number_steps):
                y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i*dt), u[:, i])
                if number_initial_states > 1:
                    if i + 1 == initial_states[count_init_states][1]:
                        x[:, i + 1] = initial_states[count_init_states][0]
                        if count_init_states < number_initial_states - 1:
                            count_init_states += 1
                    else:
                        x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i])
                else:
                    x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i])

    if system_type == 'Discrete Nonlinear':
        dt = system.dt
        number_steps = signal.number_steps
        (F, G) = system.F, system.G

        u = signal.data

        x = np.zeros([state_dimension, number_steps + 1])
        x[:, 0] = x0
        y = np.zeros([output_dimension, number_steps])
        count_init_states = 1
        for i in range(number_steps):
            y[:, i] = G(x[:, i], i * dt, u[:, i])
            if number_initial_states > 1:
                if i + 1 == initial_states[count_init_states][1]:
                    x[:, i + 1] = initial_states[count_init_states][0]
                    if count_init_states < number_initial_states - 1:
                        count_init_states += 1
                else:
                    x[:, i + 1] = F(x[:, i], i * dt, u[:, i])
            else:
                x[:, i + 1] = F(x[:, i], i * dt, u[:, i])

    if system_type == 'Continuous Linear':
        tspan = kwargs.get('tspan', np.zeros(1))
        (A, B, C, D) = system.A, system.B, system.C, system.D

        def dynamics(x, t, u):
            return np.matmul(A(t), x) + np.matmul(B(t), u(t))

        #integration_step = 0.001
        #sol = integrate(dynamics, x0, tspan, integration_step, args=(signal.u, ))
        sol = odeint(dynamics, x0, tspan, rtol=1e-13, atol=1e-13)
        x = sol.T
        y = np.zeros([output_dimension, tspan.shape[0]])
        i = 0
        for t in tspan:
            y[:, i] = np.matmul(C(t), x[:, i]) + np.matmul(D(t), u(t))
            i += 1

    if system_type == 'Continuous Nonlinear':
        tspan = kwargs.get('tspan', np.zeros(1))
        (F, G) = system.F, system.G

        #integration_step = 0.001
        #sol = integrate(F, x0, tspan, integration_step, args=(signal.u, ))
        sol = odeint(F, x0, tspan, args=(signal.u,), rtol=1e-13, atol=1e-13)
        x = sol.T
        y = np.zeros([output_dimension, tspan.shape[0]])
        i = 0
        for t in tspan:
            y[:, i] = G(x[:, i], t, signal.u(t))
            i+=1



    return (y, x)

# propagation_type = kwargs.get('propagation_type', None)

# # Propagation
# if propagation_type == 'channel':
#     y = np.zeros([output_dimension, input_dimension, number_steps])
#     for ch in range(input_dimension):
#         x = np.zeros([state_dimension, number_steps + 1])
#         x[:, 0] = x0[0]
#         uu = np.zeros([input_dimension, number_steps])
#         uu[ch, 0] = u[ch, 0]
#         for i in range(number_steps):
#             y[:, ch, i] = np.matmul(C(i*dt), x[:, i]) + np.matmul(D(i * dt), uu[:, i])
#             x[:, i + 1] = np.matmul(A(i*dt), x[:, i]) + np.matmul(B(i * dt), uu[:, i])

