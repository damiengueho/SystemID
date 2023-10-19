"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 25
"""


import numpy

def propagate_discrete_ss_model(model,
                                number_steps: int,
                                x0: numpy.ndarray = None,
                                input_data: numpy.ndarray = None):
    """
        Purpose:
            Propagate an initial condition and/or input data through a discrete-time state-space model. Model
            can be linear, bilinear or nonlinear.

        Parameters:
            - **model** (``systemID.discrete_ss_model``): the discrete-time state-space model.
            - **number_steps** (``int``): the number of steps.
            - **x0** (``numpy.ndarray``, optional): a numpy.ndarray of size (state_dimension, number_experiments)
             of initial conditions.
            - **input_data** (``numpy.ndarray``, optional): a numpy.ndarray of size
            (input_dimension, number_steps, number_experiments) of (time-varying) input data.

        Returns:
            - **y** (``numpy.ndarray``): a numpy.ndarray of size (output_dimension, number_steps, number_experiments)
             of output data.
            - **x** (``numpy.ndarray``): a numpy.ndarray of size (state_dimension, number_steps, number_experiments)
             of state data.

        Imports:
            - ``import numpy``

        Description:
            This program ...

        See Also:
            - :py:mod:`~systemID.core.functions.propagate_continuous_ss_model`
    """

    if x0 is None and input_data is None:
        raise ValueError("x0 and input_data cannot both be None")

    if x0 is None:
        if len(input_data.shape) < 3:
            number_experiments = 1
        else:
            number_experiments = input_data.shape[2]
    else:
        if len(x0.shape) < 2:
            number_experiments = 1
        else:
            number_experiments = x0.shape[1]

    if input_data is not None:
        if len(input_data.shape) == 1:
            input_data = numpy.expand_dims(input_data, axis=(0, 2))
        if len(input_data.shape) == 2:
            input_data = numpy.expand_dims(input_data, axis=2)

    if len(x0.shape) == 1:
        x0 = numpy.expand_dims(x0, axis=1)


    # Get model type and dimensions
    model_type = model.model_type
    state_dimension = model.state_dimension
    output_dimension = model.output_dimension

    # Get time parameters
    dt = model.dt

    # Type of system
    if input_data is None:
        initial_condition_response = True
    else:
        initial_condition_response = False
        u = input_data

    # Initialize vectors
    x = numpy.zeros([state_dimension, number_steps + 1, number_experiments])
    if x0 is not None:
        x[:, 0, :] = x0
    y = numpy.zeros([output_dimension, number_steps, number_experiments])

    # Get model functions
    (A, N, B, C, D, F, G) = model.A, model.N, model.B, model.C, model.D, model.F, model.G


    if model.model_type == 'linear':

        for i in range(number_steps):
            if initial_condition_response:
                y[:, i, :] = numpy.matmul(C(i * dt), x[:, i, :])
                x[:, i + 1, :] = numpy.matmul(A(i * dt), x[:, i, :])
            else:
                y[:, i, :] = numpy.matmul(C(i * dt), x[:, i, :]) + numpy.matmul(D(i * dt), u[:, i, :])
                x[:, i + 1, :] = numpy.matmul(A(i * dt), x[:, i, :]) + numpy.matmul(B(i * dt), u[:, i, :])

        return y, x


    if model_type == 'bilinear':

        for i in range(number_steps):
            if initial_condition_response:
                y[:, i] = numpy.matmul(C(i * dt), x[:, i])
                x[:, i + 1] = numpy.matmul(A(i * dt), x[:, i])

            else:
                y[:, i] = numpy.matmul(C(i * dt), x[:, i]) + numpy.matmul(D(i * dt), u[:, i])
                x[:, i + 1] = numpy.matmul(A(i * dt), x[:, i]) + numpy.matmul(N(i * dt), numpy.kron(u[:, i], x[:, i])) + numpy.matmul(B(i * dt), u[:, i])

        return y, x


    if model_type == 'nonlinear':

        for i in range(number_steps):
                y[:, i] = G(x[:, i], i * dt, u[:, i])
                x[:, i + 1] = F(x[:, i], i * dt, u[:, i])

        return y, x



def propagate_continuous_ss_model(model,
                                  tspan: numpy.ndarray,
                                  x0: numpy.ndarray = None,
                                  input_data: numpy.ndarray = None):
    """
        Purpose:
            Propagate an initial condition and/or input data through a continuous-time state-space model. Model
            can be linear, bilinear or nonlinear.

    Parameters:
        - **model** (``systemID.continuous_ss_model``): the continuous-time state-space model.
        - **tspan** (``numpy.ndarray``): a numpy.ndarray that represents the time span.
        - **x0** (``numpy.ndarray``, optional): a numpy.ndarray of size (state_dimension, number_experiments)
         of initial conditions.
        - **input_data** (``func``, optional): a function that represents the input data.

    Returns:
        - **y** (``numpy.ndarray``): a numpy.ndarray of size (output_dimension, number_steps, number_experiments)
         of output data.
        - **x** (``numpy.ndarray``): a numpy.ndarray of size (state_dimension, number_steps, number_experiments)
         of state data.

    Imports:
        - ``import numpy``

    Description:
        This program ...

    See Also:
        - :py:mod:`~systemID.core.functions.propagate_discrete_ss_model`
    """