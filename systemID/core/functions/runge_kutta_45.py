"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 25
"""


import numpy

def runge_kutta_45(dynamics,
                   x0: numpy.ndarray,
                   tspan: numpy.ndarray,
                   integration_step: float,
                   args = None,
                   decimals: int = 8):
    """
        Purpose:
            Runge-Kutta 45 integration scheme. Used in replacement of **scipy.integrate.odeint** or other
            numerical integration schemes if the one wants to keep the integration step **integration_step**
            constant.

        Parameters:
            - **dynamics** (``fun``): dynamics equations.
            - **order** (``int``): the order of single monomials to be appended.

        Returns:
            - **output** (``numpy.ndarray``): the integrated signal.

        Imports:
            - ``import numpy``

        Description:
            abc

        See Also:
            -
    """


    # Number of iterations
    n = int((round(tspan[-1] / integration_step)))

    # Input
    u = args[0]

    # Initialization
    output = numpy.zeros([len(tspan), len(x0)])
    output[0, :] = x0
    x = x0
    t = 0
    j = 1

    # Iterate for number of iterations
    for i in range(1, n+1):
        # print('t =', t)
        k1 = integration_step * dynamics(x, t, u)
        half_t = numpy.round(t + 0.5 * integration_step, decimals=decimals)
        k2 = integration_step * dynamics(x + 0.5 * k1, half_t, u)
        k3 = integration_step * dynamics(x + 0.5 * k2, half_t, u)
        next_t = numpy.round(t + integration_step, decimals=decimals)
        k4 = integration_step * dynamics(x + k3, next_t, u)

        # Update next value of x
        x = x + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if i % int(round(n / (len(tspan) - 1))) == 0:
            output[j, :] = x
            j = j + 1

        # Update next value of t
        t = next_t

    return output
