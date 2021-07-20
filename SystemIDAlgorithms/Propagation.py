"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 11
Date: July 2021
Python: 3.7.7
"""


import numpy as np
from scipy.integrate import odeint

from SystemIDAlgorithms.Integrate import integrate


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

        x = np.zeros([state_dimension, number_steps + 1])
        x[:, 0] = x0
        y = np.zeros([output_dimension, number_steps])
        count_init_states = 1

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


    return (y, x)
