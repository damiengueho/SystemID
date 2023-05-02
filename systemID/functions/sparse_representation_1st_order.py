"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""


import numpy as np
import scipy.linalg as LA
import cvxpy as cp
from scipy.integrate import odeint

from systemID.functions import interpolant_functions


def sparse_representation_1st_order(input_signals, output_signals, basis_functions, filter_coefficient, relax_coefficient, threshold, max_iterations, **kwargs):
    """
        Purpose:
            a

        Parameters:
            - **input_signals** (``list``):

        Returns:
            - **A** (``fun``): a

        Imports:
            - ``import numpy as np``
            - ``import scipy.linalg as LA``

        Description:
            We describe.

        See Also:
            - :py:mod:``
        """

    ## Lengths input_signals and output_signals must be the same
    number_signals = len(output_signals)
    ## total_time and number_steps for both must be the same

    # Create interpolant functions for input signals
    interpolant_inputs = []
    for s in input_signals:
        tspan = np.linspace(0, s.total_time, s.number_steps)
        interpolant_inputs.append(interpolant_functions([tspan], [s], b_spline_degree=3)[0])

    # Create interpolant functions for output signals
    number_steps = []
    interpolant_outputs = []
    tspans = []
    for s in output_signals:
        number_steps.append(s.number_steps)
        tspan = np.linspace(0, s.total_time, s.number_steps)
        tspans.append(tspan)
        interpolant_outputs.append(interpolant_functions([tspan], [s], b_spline_degree=3)[0])

    #
    output_dimension = output_signals[0].dimension
    number_basis_functions = len(basis_functions)
    total_number_steps = sum(number_steps)

    # Initialize coefficient vectors
    coefficients_least_squares = np.zeros([number_basis_functions, output_dimension])
    coefficients_sparse_intermediate = np.zeros([number_basis_functions, output_dimension])
    coefficients_sparse = np.zeros([number_basis_functions, output_dimension])

    # Initialize matrices
    Y1 = np.zeros([output_dimension, total_number_steps])
    U = np.zeros([output_dimension, total_number_steps])
    PHI = np.zeros([total_number_steps, number_basis_functions, output_dimension])
    Xt = np.zeros([total_number_steps, output_dimension])
    C = np.zeros([output_dimension, number_basis_functions, max_iterations])

    ## Integration of the N+3 equations
    for k in range(output_dimension):
        print('Dimension ', k + 1, ' of ', output_dimension)

        ct = 0
        for s in range(number_signals):
            print('Signal number ', s + 1, ' of ', number_signals)

            def dynamics(X, t):

                dXdt = np.zeros([2 + number_basis_functions])

                x = interpolant_outputs[s](t)
                u = interpolant_inputs[s](t)

                dXdt[0] = -filter_coefficient * X[0] - filter_coefficient * x[k]
                dXdt[1] = -filter_coefficient * X[1] + u[k]

                for i in range(number_basis_functions):
                    dXdt[2 + i] = -filter_coefficient * X[2 + i] + basis_functions[i](x)

                return dXdt

            # Solve Differential Equation
            y1_0 = -output_signals[s].data[k, 0]
            u_0 = 0
            Phi_0 = np.zeros([1, number_basis_functions])
            X0 = np.concatenate((np.array([[y1_0, u_0]]), Phi_0), axis=1)

            X = odeint(dynamics, X0[0, :], tspans[s], rtol=1e-13, atol=1e-13)

            y1 = X[:, 0:1]
            u = X[:, 1:2]
            Phi = X[:, 2:2 + number_basis_functions]

            # Define xf, xt and Phi
            xf = np.transpose(output_signals[s].data[k:k + 1, :]) + y1
            xt = xf - u
            Y1[k:k + 1, ct:ct + number_steps[s]] = np.transpose(y1)
            U[k:k + 1, ct:ct + number_steps[s]] = np.transpose(u)
            PHI[ct:ct + number_steps[s], :, k] = Phi
            Xt[ct:ct + number_steps[s], k:k + 1] = xt
            ct = ct + number_steps[s]

        # Least Square Solution
        theta = np.matmul(LA.pinv(PHI[:, :, k]), Xt[:, k:k + 1])
        coefficients_least_squares[:, k:k + 1] = theta

        # Sparse solution
        H = PHI[:, :, k]
        it = 0

        W = np.diag(np.ones(number_basis_functions))
        init_weight = kwargs.get('init_weight', 'N/A')
        if init_weight == 'least_squares':
            for i in range(number_basis_functions):
                W[i, i] = 1 / (np.abs(theta[i, 0]) + 1e-12)
                W = W / (np.max(np.abs(np.diag(W))))

        while it < max_iterations:
            print('Iteration: ', it)
            c = cp.Variable(shape=H.shape[1])
            objective = cp.Minimize(cp.norm(W @ c, 1))
            constraints = [cp.norm(Xt[:, k] - H @ c, 2) <= relax_coefficient * cp.norm(Xt[:, k] - np.matmul(H, theta)[:, 0], 2)]
            prob = cp.Problem(objective, constraints)
            prob.solve()
            # prob.solve(verbose=True)

            C[k, :, it] = c.value

            for i in range(H.shape[1]):
                W[i, i] = 1 / (np.abs(c.value[i]) + 1e-12)
            W = W / (np.max(np.abs(np.diag(W))))

            it = it + 1

        indices_non0 = [i for i, value in enumerate(c.value) if np.abs(value) >= threshold]
        indices_0 = [i for i, value in enumerate(c.value) if np.abs(value) < threshold]

        H_sparse = np.take(H, indices_non0, axis=1)
        theta_sparse = np.matmul(LA.pinv(H_sparse), Xt[:, k:k + 1])

        count = 0
        for i in range(number_basis_functions):
            if count < len(indices_non0):
                if indices_non0[count] == i:
                    coefficients_sparse_intermediate[i, k] = theta_sparse[count, 0]
                    count = count + 1

        indices_non0_final = [i for i, value in enumerate(list(coefficients_sparse_intermediate[:, k])) if np.abs(value) >= threshold]
        indices_0_final = [i for i, value in enumerate(list(coefficients_sparse_intermediate[:, k])) if np.abs(value) < threshold]

        H_sparse = np.take(H, indices_non0_final, axis=1)
        theta_sparse_final = np.matmul(LA.pinv(H_sparse), Xt[:, k:k + 1])

        print('indices non zero:', indices_non0_final)
        print('indices zero:', indices_0_final)

        count = 0
        for i in range(number_basis_functions):
            if count < len(indices_non0_final):
                if indices_non0_final[count] == i:
                    coefficients_sparse[i, k] = theta_sparse_final[count, 0]
                    count = count + 1

    return coefficients_least_squares, coefficients_sparse
