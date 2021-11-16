"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""



import numpy as np


def integrate(dynamics, x0, tspan, integration_step, **kwargs):


    # Number of iterations
    n = int((round(tspan[-1] / integration_step)))

    # Input
    args = kwargs.get('args', False)
    u = args[0]

    # Initialization
    output = np.zeros([len(tspan), len(x0)])
    output[0, :] = x0
    x = x0
    t = 0
    j = 1

    # Iterate for number of iterations
    for i in range(1, n+1):
        k1 = integration_step * dynamics(x, t, u)
        half_t = np.round(t + 0.5 * integration_step, decimals=4)
        k2 = integration_step * dynamics(x + 0.5 * k1, half_t, u)
        k3 = integration_step * dynamics(x + 0.5 * k2, half_t, u)
        next_t = np.round(t + integration_step, decimals=3)
        k4 = integration_step * dynamics(x + k3, next_t, u)

        # Update next value of x
        x = x + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if i % int(round(n / (len(tspan) - 1))) == 0:
            output[j, :] = x
            j = j + 1

        # Update next value of t
        t = next_t


    return output