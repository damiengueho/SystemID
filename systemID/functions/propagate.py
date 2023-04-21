"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""


import numpy as np
from scipy.integrate import odeint

from systemID.signals.discrete import discrete_signal
from systemID.functions.runge_kutta_45 import runge_kutta_45
from systemID.functions.state_transition_tensor_propagation import state_transition_tensor_propagation


def propagate(signal, system, **kwargs):
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

    # Get system's parameters
    state_dimension = system.state_dimension
    output_dimension = system.output_dimension
    x0 = system.x0

    # Propagate depending on system type
    system_type = system.system_type

    if system_type == 'discrete linear':
        dt = system.dt
        number_steps = signal.number_steps
        (A, B, C, D) = system.A, system.B, system.C, system.D

        u = signal.data

        K = kwargs.get('feedback_matrix', None)
        if K is not None:
            feedback_signal = kwargs.get('feedback_signal', np.zeros([output_dimension, number_steps]))

        process_noise_signal = kwargs.get('process_noise_signal', None)
        measurement_noise_signal = kwargs.get('measurement_noise_signal', None)

        x = np.zeros([state_dimension, number_steps + 1])
        x[:, 0] = x0
        y = np.zeros([output_dimension, number_steps])

        if K is None:
            if process_noise_signal is None:
                if measurement_noise_signal is None:
                    for i in range(number_steps):
                        y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i * dt), u[:, i])
                        x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i])
                else:
                    for i in range(number_steps):
                        y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i * dt), u[:, i]) + measurement_noise_signal.data[:, i]
                        x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i])
            else:
                if measurement_noise_signal is None:
                    for i in range(number_steps):
                        y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i * dt), u[:, i])
                        x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i]) + process_noise_signal.data[:, i]
                else:
                    for i in range(number_steps):
                        y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i * dt), u[:, i]) + measurement_noise_signal.data[:, i]
                        x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i]) + process_noise_signal.data[:, i]
        else:
            if process_noise_signal is None:
                if measurement_noise_signal is None:
                    for i in range(number_steps):
                        y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i * dt), u[:, i]) + np.matmul(K(i * dt), feedback_signal.data[:, i])
                        x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i])
                else:
                    for i in range(number_steps):
                        y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i * dt), u[:, i]) + measurement_noise_signal.data[:, i] + np.matmul(K(i * dt), feedback_signal.data[:, i])
                        x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i])
            else:
                if measurement_noise_signal is None:
                    for i in range(number_steps):
                        y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i * dt), u[:, i]) + np.matmul(K(i * dt), feedback_signal.data[:, i])
                        x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i]) + process_noise_signal.data[:, i]
                else:
                    for i in range(number_steps):
                        y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i * dt), u[:, i]) + measurement_noise_signal.data[:, i] + np.matmul(K(i * dt), feedback_signal.data[:, i])
                        x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i]) + process_noise_signal.data[:, i]

        output_signal = discrete_signal(frequency=signal.frequency, data=y)
        state = discrete_signal(frequency=signal.frequency, data=x)



    if system_type == 'discrete bilinear':
        dt = system.dt
        number_steps = signal.number_steps
        (A, N, B, C, D) = system.A, system.N, system.B, system.C, system.D

        u = signal.data

        x = np.zeros([state_dimension, number_steps + 1])
        x[:, 0] = x0
        y = np.zeros([output_dimension, number_steps])

        for i in range(number_steps):
            y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i * dt), u[:, i])
            x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(N(i * dt), np.kron(u[:, i], x[:, i])) + np.matmul(B(i * dt), u[:, i])

        output_signal = discrete_signal(frequency=signal.frequency, data=y)
        state = discrete_signal(frequency=signal.frequency, data=x)




    if system_type == 'discrete nonlinear':
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

        for i in range(number_steps):
            if measurement_noise:
                y[:, i] = G(x[:, i], i * dt, u[:, i]) + measurement_noise_signal.u(i * dt)
            else:
                y[:, i] = G(x[:, i], i * dt, u[:, i])

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

        output_signal = discrete_signal(frequency=signal.frequency, data=y)
        state = discrete_signal(frequency=signal.frequency, data=x)

    if system_type == 'continuous linear':
        tspan = kwargs.get('tspan', np.zeros(1))
        (A, B, C, D) = system.A, system.B, system.C, system.D

        def dynamics(x, t, u):
            return np.matmul(A(t), x) + np.matmul(B(t), u(t))

        fixed_step_size = kwargs.get('fixed_step_size', False)
        if fixed_step_size:
            integration_step = kwargs.get('integration_step', 0.001)
            sol = runge_kutta_45(dynamics, x0, tspan, integration_step, args=(signal.u, ))
        else:
            sol = odeint(dynamics, x0, tspan, args=(signal.u, ), rtol=1e-12, atol=1e-12)

        x = sol.T
        y = np.zeros([output_dimension, tspan.shape[0]])
        i = 0
        for t in tspan:
            y[:, i] = np.matmul(C(t), x[:, i]) + np.matmul(D(t), signal.u(t))
            i += 1

        output_signal = discrete_signal(frequency=1 / (tspan[1] - tspan[0]), data=y)
        state = discrete_signal(frequency=1 / (tspan[1] - tspan[0]), data=x)

    if system_type == 'continuous bilinear':
        tspan = kwargs.get('tspan', np.zeros(1))
        (A, N, B, C, D) = system.A, system.N, system.B, system.C, system.D

        def dynamics(x, t, u):
            return np.matmul(A(t), x) + np.matmul(N(t), np.kron(u(t), x)) + np.matmul(B(t), u(t))

        fixed_step_size = kwargs.get('fixed_step_size', False)
        if fixed_step_size:
            integration_step = kwargs.get('integration_step', 0.001)
            sol = runge_kutta_45(dynamics, x0, tspan, integration_step, args=(signal.u, ))
        else:
            sol = odeint(dynamics, x0, tspan, args=(signal.u, ), rtol=1e-12, atol=1e-12)

        x = sol.T
        y = np.zeros([output_dimension, tspan.shape[0]])
        i = 0
        for t in tspan:
            y[:, i] = np.matmul(C(t), x[:, i]) + np.matmul(D(t), signal.u(t))
            i += 1

        output_signal = discrete_signal(frequency=1 / (tspan[1] - tspan[0]), data=y)
        state = discrete_signal(frequency=1 / (tspan[1] - tspan[0]), data=x)


    if system_type == 'continuous nonlinear':
        tspan = kwargs.get('tspan', np.zeros(1))
        (F, G) = system.F, system.G

        fixed_step_size = kwargs.get('fixed_step_size', False)
        if fixed_step_size:
            integration_step = kwargs.get('integration_step', 0.001)
            sol = runge_kutta_45(F, x0, tspan, integration_step, args=(signal.u, ))
        else:
            sol = odeint(F, x0, tspan, args=(signal.u,), rtol=1e-12, atol=1e-12)

        x = sol.T
        y = np.zeros([output_dimension, tspan.shape[0]])
        i = 0
        for t in tspan:
            y[:, i] = G(x[:, i], t, signal.u(t))
            i += 1

        output_signal = discrete_signal(frequency=1 / (tspan[1] - tspan[0]), data=y)
        state = discrete_signal(frequency=1 / (tspan[1] - tspan[0]), data=x)


    if system_type == 'continuous higher order':

        tspan = kwargs.get('tspan', np.zeros(1))
        number_steps = len(tspan)

        A_vec, _ = state_transition_tensor_propagation(system.sensitivities, system.F, system.u_nominal, system.x0_nominal, tspan)

        G = system.G

        x = np.zeros([state_dimension, number_steps])
        x[:, 0] = x0
        y = np.zeros([output_dimension, number_steps])

        if system.order == 1:
            for i in range(number_steps):
                A1 = A_vec[i, state_dimension:state_dimension + state_dimension ** 2].reshape(state_dimension, state_dimension)
                x[:, i] = np.matmul(A1, x0)
                y[:, i] = G(x[:, i], tspan[i], signal.u(tspan[i]))

        if system.order == 2:
            for i in range(number_steps):
                A1 = A_vec[i, state_dimension:state_dimension + state_dimension ** 2].reshape(state_dimension, state_dimension)
                A2 = A_vec[i, state_dimension + state_dimension ** 2:].reshape(state_dimension, state_dimension, state_dimension)
                x[:, i] = np.matmul(A1, x0) + (1 / 2) * np.tensordot(A2, np.tensordot(x0, x0, axes=0), axes=([1, 2], [0, 1]))
                y[:, i] = G(x[:, i], tspan[i], signal.u(tspan[i]))

        if system.order == 3:
            for i in range(number_steps):
                A1 = A_vec[i, state_dimension:state_dimension + state_dimension ** 2].reshape(state_dimension, state_dimension)
                A2 = A_vec[i, state_dimension + state_dimension ** 2:state_dimension + state_dimension ** 2 + state_dimension ** 3].reshape(state_dimension, state_dimension, state_dimension)
                A3 = A_vec[i, state_dimension + state_dimension ** 2 + state_dimension ** 3:].reshape(state_dimension, state_dimension, state_dimension, state_dimension)
                x[:, i] = np.matmul(A1, x0) + (1 / 2) * np.tensordot(A2, np.tensordot(x0, x0, axes=0), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(A3, np.tensordot(np.tensordot(x0, x0, axes=0), x0, axes=0), axes=([1, 2, 3], [0, 1, 2]))
                y[:, i] = G(x[:, i], tspan[i], signal.u(tspan[i]))

        if system.order == 4:
            for i in range(number_steps):
                A1 = A_vec[i, state_dimension:state_dimension + state_dimension ** 2].reshape(state_dimension, state_dimension)
                A2 = A_vec[i, state_dimension + state_dimension ** 2:state_dimension + state_dimension ** 2 + state_dimension ** 3].reshape(state_dimension, state_dimension, state_dimension)
                A3 = A_vec[i, state_dimension + state_dimension ** 2 + state_dimension ** 3:state_dimension + state_dimension ** 2 + state_dimension ** 3 + state_dimension ** 4].reshape(state_dimension, state_dimension, state_dimension, state_dimension)
                A4 = A_vec[i, state_dimension + state_dimension ** 2 + state_dimension ** 3 + state_dimension ** 4:].reshape(state_dimension, state_dimension, state_dimension, state_dimension, state_dimension)
                x[:, i] = np.matmul(A1, x0) + (1 / 2) * np.tensordot(A2, np.tensordot(x0, x0, axes=0), axes=([1, 2], [0, 1])) + (1 / 6) * np.tensordot(A3, np.tensordot(np.tensordot(x0, x0, axes=0), x0, axes=0), axes=([1, 2, 3], [0, 1, 2])) + (1 / 24) * np.tensordot(A4, np.tensordot(np.tensordot(np.tensordot(x0, x0, axes=0), x0, axes=0), x0, axes=0), axes=([1, 2, 3, 4], [0, 1, 2, 3]))
                y[:, i] = G(x[:, i], tspan[i], signal.u(tspan[i]))

        output_signal = discrete_signal(frequency=1 / (tspan[1] - tspan[0]), data=y)
        state = discrete_signal(frequency=1 / (tspan[1] - tspan[0]), data=x)


    return output_signal, state
