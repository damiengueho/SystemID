"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""


import numpy as np
import scipy.linalg as LA

from systemID.ClassesGeneral.ClassSignal import DiscreteSignal




def prediction(nominal_reference, system_reference, nominal, system, input_signal, starting_step, **kwargs):
    """
    Purpose:


    Parameters:
        -

    Returns:
        -

    Imports:
        -

    Description:


    See Also:
        -
    """

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
        (A, B, C, D) = system.A, system.B, system.C, system.D
        number_steps = input_signal.number_steps
        x = np.zeros([state_dimension, number_steps + 1])
        x[:, 0] = x0
        y = np.zeros([output_dimension, number_steps])
        count_init_states = 1

        u = input_signal.data

        indexes = np.zeros(number_steps)

        selected_for_propagation = {"A": [], 'B': [], 'C': [], 'D': [], 'indexes': np.zeros(number_steps)}

        # Propagation on training data
        for i in range(starting_step):
            indexes[i] = i
            y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i * dt), u[:, i])
            selected_for_propagation['C'].append(C(i * dt))
            selected_for_propagation['D'].append(D(i * dt))
            selected_for_propagation['indexes'][i] = i
            if number_initial_states > 1:
                if i + 1 == initial_states[count_init_states][1]:
                    x[:, i + 1] = initial_states[count_init_states][0]
                    if count_init_states < number_initial_states - 1:
                        count_init_states += 1
                else:
                    x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i])
                    selected_for_propagation['A'].append(A(i * dt))
                    selected_for_propagation['B'].append(B(i * dt))

            else:
                x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i])
                selected_for_propagation['A'].append(A(i * dt))
                selected_for_propagation['B'].append(B(i * dt))

        # Prediction
        left_window_size = kwargs.get('left_window_size', 0)
        right_window_size = kwargs.get('right_window_size', 0)
        for i in range(starting_step, number_steps):
            current_left_window_size = min(i, left_window_size)
            current_right_window_size = min(number_steps - i, right_window_size)
            nearest_index = min(range(current_left_window_size, nominal_reference.number_steps - current_right_window_size), key=lambda j: LA.norm(nominal.data[:, i - current_left_window_size:i + current_right_window_size] - nominal_reference.data[:, j - current_left_window_size:j + current_right_window_size]))

            selected_for_propagation['indexes'][i] = nearest_index

            C_mat = system_reference.C((nearest_index) * dt)
            D_mat = system_reference.D((nearest_index) * dt)
            selected_for_propagation['C'].append(C_mat)
            selected_for_propagation['D'].append(D_mat)

            y[:, i] = np.matmul(C_mat, x[:, i]) + np.matmul(D_mat, u[:, i])

            A_mat = system_reference.A((nearest_index) * dt)
            B_mat = system_reference.B((nearest_index) * dt)
            selected_for_propagation['A'].append(A_mat)
            selected_for_propagation['B'].append(B_mat)

            if number_initial_states > 1:
                if i + 1 == initial_states[count_init_states][1]:
                    x[:, i + 1] = initial_states[count_init_states][0]
                    if count_init_states < number_initial_states - 1:
                        count_init_states += 1
                else:
                    x[:, i + 1] = np.matmul(A_mat, x[:, i]) + np.matmul(B_mat, u[:, i])
            else:
                x[:, i + 1] = np.matmul(A_mat, x[:, i]) + np.matmul(B_mat, u[:, i])



        # Building output signal
        output_signal_predicted = DiscreteSignal(output_dimension, 'Output Signal Predicted', input_signal.total_time, input_signal.frequency, signal_shape='External', data=y)



    return selected_for_propagation, output_signal_predicted
















def prediction2(nominal_state_training, nominal_input_training, nominal_state_prediction, nominal_input_prediction, system, input_signal):


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
        number_steps = input_signal.number_steps
        number_steps_training = nominal_input_training.number_steps
        #number_steps_prediction = nominal_input_prediction.number_steps
        (A, B, C, D) = system.A, system.B, system.C, system.D

        x = np.zeros([state_dimension, number_steps + 1])
        x[:, 0] = x0
        print(x0)
        y = np.zeros([output_dimension, number_steps])
        count_init_states = 1

        u = input_signal.data

        # Build Look-up Table
        A_list = []
        B_list = []
        C_list = []
        D_list = []
        A_chosen = []
        indexes = np.zeros(number_steps)
        nominal_state_history = []
        nominal_input_history = []
        for i in range(number_steps_training):
            A_list.append(A(i * dt))
            B_list.append(B(i * dt))
            C_list.append(C(i * dt))
            D_list.append(D(i * dt))
            nominal_state_history.append(nominal_state_training.data[:, i])
            nominal_input_history.append(nominal_input_training.data[:, i])


        # Propagation on training data
        for i in range(number_steps_training):
            indexes[i] = i
            y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i * dt), u[:, i])
            if number_initial_states > 1:
                if i + 1 == initial_states[count_init_states][1]:
                    x[:, i + 1] = initial_states[count_init_states][0]
                    if count_init_states < number_initial_states - 1:
                        count_init_states += 1
                else:
                    x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i])
                    A_chosen.append(A(i * dt))
            else:
                x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i])
                A_chosen.append(A(i * dt))


        # Prediction
        nearest_index = 0
        for i in range(nominal_input_training.number_steps, number_steps):
            print('i =', i)
            nearest_index = min(range(len(nominal_state_history)), key=lambda j: (LA.norm(nominal_state_history[j] - nominal_state_prediction.data[:, i - nominal_input_training.number_steps]) + LA.norm(nominal_input_history[j] - nominal_input_prediction.data[:, i - nominal_input_training.number_steps])))
            indexes[i] = nearest_index
            C_mat = C_list[nearest_index]
            D_mat = D_list[nearest_index]
            y[:, i] = np.matmul(C_mat, x[:, i]) + np.matmul(D_mat, u[:, i])
            A_mat = A_list[nearest_index]
            A_chosen.append(A_mat)
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


        # Building Signals
        output_signal_predicted = DiscreteSignal(output_dimension, 'Output Signal Predicted', input_signal.total_time, nominal_input_training.frequency, signal_shape='External', data=y)



    return A_list, B_list, C_list, D_list, A_chosen, indexes, nominal_state_history, nominal_input_history, x, y, output_signal_predicted





#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# def prediction2(nominal_state, nominal_input, system, nominal_state_reference, nominal_input_reference, system_reference, input_signal, number_steps_not_predicted, number_steps_predicted):
#
#
#     # Get general parameters of the system
#     state_dimension = system.state_dimension
#     output_dimension = system.output_dimension
#     x0 = system.x0
#     initial_states = system.initial_states
#     number_initial_states = len(initial_states)
#
#
#     # Propagate depending on system type
#     system_type = system.system_type
#
#     if system_type == 'Discrete Linear':
#         dt = system.dt
#         number_steps = number_steps_not_predicted + number_steps_predicted
#         (A, B, C, D) = system.A, system.B, system.C, system.D
#
#         x = np.zeros([state_dimension, number_steps + 1])
#         x[:, 0] = x0
#         print(x0)
#         y = np.zeros([output_dimension, number_steps])
#         count_init_states = 1
#
#         u = input_signal.data
#
#         # Build Look-up Table
#         A_list = []
#         B_list = []
#         C_list = []
#         D_list = []
#         A_chosen = []
#         indexes = np.zeros(number_steps)
#         nominal_state_history = []
#         nominal_input_history = []
#         for i in range(nominal_state_reference.number_steps):
#             A_list.append(system_reference.A(i * dt))
#             B_list.append(system_reference.B(i * dt))
#             C_list.append(system_reference.C(i * dt))
#             D_list.append(system_reference.D(i * dt))
#             nominal_state_history.append(nominal_state_reference.data[:, i])
#             nominal_input_history.append(nominal_input_reference.data[:, i])
#
#
#         # Propagation on training data
#         for i in range(number_steps_not_predicted):
#             indexes[i] = i
#             y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i * dt), u[:, i])
#             if number_initial_states > 1:
#                 if i + 1 == initial_states[count_init_states][1]:
#                     x[:, i + 1] = initial_states[count_init_states][0]
#                     if count_init_states < number_initial_states - 1:
#                         count_init_states += 1
#                 else:
#                     x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i])
#                     A_chosen.append(A(i * dt))
#             else:
#                 x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i])
#                 A_chosen.append(A(i * dt))
#
#
#         # Prediction
#         nearest_index = 0
#         for i in range(number_steps_predicted):
#             print('i =', i)
#             nearest_index = min(range(len(nominal_state_history)), key=lambda j: (LA.norm(nominal_state_history[j] - nominal_state.data[:, i + number_steps_not_predicted]) + LA.norm(nominal_input_history[j] - nominal_input.data[:, i + number_steps_not_predicted])))
#             indexes[i] = nearest_index
#             C_mat = C_list[nearest_index]
#             D_mat = D_list[nearest_index]
#             y[:, i] = np.matmul(C_mat, x[:, i]) + np.matmul(D_mat, u[:, i])
#             A_mat = A_list[nearest_index]
#             A_chosen.append(A_mat)
#             B_mat = B_list[nearest_index]
#             if number_initial_states > 1:
#                 if i + 1 == initial_states[count_init_states][1]:
#                     x[:, i + 1] = initial_states[count_init_states][0]
#                     if count_init_states < number_initial_states - 1:
#                         count_init_states += 1
#                 else:
#                     x[:, i + 1] = np.matmul(A_mat, x[:, i]) + np.matmul(B_mat, u[:, i])
#             else:
#                 x[:, i + 1] = np.matmul(A_mat, x[:, i]) + np.matmul(B_mat, u[:, i])
#
#
#         # Building Signals
#         output_signal_predicted = DiscreteSignal(output_dimension, 'Output Signal Predicted', input_signal.total_time, input_signal.frequency, signal_shape='External', data=y)
#
#
#
#     return A_list, B_list, C_list, D_list, A_chosen, indexes, nominal_state_history, nominal_input_history, x, y, output_signal_predicted