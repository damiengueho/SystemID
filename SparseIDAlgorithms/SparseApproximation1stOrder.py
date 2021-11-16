"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


import numpy as np
import cvxpy as cp
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import numpy.linalg as LA

from SparseIDAlgorithms.GeneratePolynomialIndex import generatePolynomialIndex
from SparseIDAlgorithms.GeneratePolynomialBasisFunctions import generatePolynomialBasisFunctions
from ClassesGeneral.ClassSignal import DiscreteSignal, subtract2Signals


def sparseApproximation1stOrder(signals, input_signals, x0s_testing, input_signals_testing, order, max_order, post_treatment, l1, alpha, delta, epsilon, max_iterations, shift):

    # Get signals parameters - All signals have same dimension
    number_signals = len(signals)
    dimension = signals[0].dimension
    frequencies = []
    total_times = []
    numbers_steps = []
    tspans = []
    data = []
    interp_data = []
    inputs = []
    interp_inputs = []
    interp_inputs_testing = []

    for s in range(number_signals):
        frequencies.append(signals[s].frequency)
        total_times.append(signals[s].total_time)
        numbers_steps.append(signals[s].number_steps)
        tspan = np.linspace(0, signals[s].total_time, signals[s].number_steps)
        tspans.append(tspan)
        data.append(signals[s].data)
        interp_data.append(interp1d(tspan, signals[s].data, kind='cubic'))
        inputs.append(input_signals[s].data)
        interp_inputs.append(interp1d(tspan, input_signals[s].data, kind='cubic'))

    for s in range(len(input_signals_testing)):
        tspan = np.linspace(0, input_signals_testing[s].total_time, input_signals_testing[s].number_steps)
        interp_inputs_testing.append(interp1d(tspan, input_signals_testing[s].data, kind='cubic'))

    for s in range(number_signals):
        total_times[s] = total_times[s] - shift
        numbers_steps[s] = int(numbers_steps[s] - shift * frequencies[s])
        tspans[s] = np.linspace(0, total_times[s], numbers_steps[s])


    # Create Index and Basis functions
    index = generatePolynomialIndex(dimension, order, post_treatment, max_order)
    index_length, _ = index.shape
    basis_functions = generatePolynomialBasisFunctions(dimension, index)


    ## Initialize Coefficient Vectors
    THETA_LS = np.zeros([index_length, dimension])
    THETA_SPARSE = np.zeros([index_length, dimension])
    Y1 = np.zeros([dimension, sum(numbers_steps)])
    U = np.zeros([dimension, sum(numbers_steps)])
    PHI = np.zeros([sum(numbers_steps), index_length, dimension])
    Xt = np.zeros([sum(numbers_steps), dimension])
    C = np.zeros([dimension, index_length, max_iterations])


    ## Integration of the N+3 equations
    for k in range(dimension):
        print('Dimension ', k + 1, ' of ', dimension)

        ct = 0
        for s in range(number_signals):
            print('Signal number ', s + 1, ' of ', number_signals)

            def Dynamics(X, t):

                dXdt = np.zeros([2 + index_length])

                x = interp_data[s](t)
                u = interp_inputs[s](t)

                dXdt[0] = -l1*X[0] - l1*x[k]
                dXdt[1] = -l1*X[1] + u[k]

                for i in range(index_length):
                    dXdt[2 + i] = -l1*X[2 + i] + basis_functions[i](x)

                return dXdt

            # Solve Differential Equation
            y1_0 = -data[s][k, 0]
            u_0 = 0
            Phi_0 = np.zeros([1, index_length])
            X0 = np.concatenate((np.array([[y1_0, u_0]]), Phi_0), axis=1)
            X = odeint(Dynamics, X0[0, :], tspans[s], rtol=1e-13, atol=1e-13)
            y1 = X[:, 0:1]
            u = X[:, 1:2]
            Phi = X[:, 2:2 + index_length]

            # Define xf, xt and Phi
            xf = np.transpose(data[s][k:k + 1, 0:numbers_steps[s]]) + y1
            xt = xf - u
            Y1[k:k + 1, ct:ct+numbers_steps[s]] = np.transpose(y1)
            U[k:k + 1, ct:ct+numbers_steps[s]] = np.transpose(u)
            PHI[ct:ct+numbers_steps[s], :, k] = Phi
            Xt[ct:ct+numbers_steps[s], k:k+1] = xt
            ct = ct + numbers_steps[s]


        # Least Square Solution
        theta = np.matmul(LA.pinv(PHI[:, :, k]), Xt[:, k:k+1])
        #theta = np.matmul(np.matmul(LA.inv(np.matmul(PHI[:, :, k].T, PHI[:, :, k])), PHI[:, :, k].T), Xt[:, k:k+1])
        THETA_LS[:, k:k+1] = theta
        print(THETA_LS)

        # Sparse solution
        H = PHI[:, :, k]
        it = 0

        W = np.diag(np.ones(H.shape[1]))
        for i in range(H.shape[1]):
            W[i, i] = 1 / (np.abs(theta[i, 0]) + epsilon)
        W = W / (np.max(np.abs(np.diag(W))) * 0.8 * 1e-1)

        while it < max_iterations:
            print('Iteration: ', it)
            c = cp.Variable(shape=H.shape[1])
            objective = cp.Minimize(cp.norm(W * c, 1))
            constraints = [cp.norm(Xt[:, k] - H * c, 2) <= alpha * cp.norm(Xt[:, k] - np.matmul(H, theta)[:, 0], 2)]
            prob = cp.Problem(objective, constraints)
            prob.solve(verbose=True)
            print('c', c.value)
            C[k, :, it] = c.value

            for i in range(H.shape[1]):
                W[i, i] = 1 / (np.abs(c.value[i]) + epsilon)
            W = W / (np.max(np.abs(np.diag(W))) * 0.8 * 1e-1)
            #print('W', W)

            it = it + 1

        index_non0 = []
        for i in range(H.shape[1]):
            index_non0.append(i)
        index_0 = []
        ind = []
        w = []
        for i in range(H.shape[1]):
            if np.abs(c.value[i]) < delta[k]:
                index_0.append(index_non0[i])
                ind.append(i)
            if np.abs(c.value[i]) >= delta[k]:
                w.append(c.value[i])

        ind.reverse()
        for i in range(len(ind)):
            del index_non0[ind[i]]

        print('index_non0', index_non0)
        print('index_0', index_0)

        H_sparse = np.take(H, index_non0, axis=1)
        theta_sparse = np.matmul(LA.pinv(H_sparse), Xt[:, k:k+1])

        count = 0
        for i in range(index_length):
            if count < len(index_non0):
                if index_non0[count] == i:
                    THETA_SPARSE[i, k] = theta_sparse[count, 0]
                    count = count + 1

    #
    # THETA_SPARSE = np.zeros([70, 4])
    # THETA_SPARSE[15, 0] = 1.000000
    # THETA_SPARSE[35, 1] = 1.000000
    # THETA_SPARSE[1, 2] = 889.55136
    # THETA_SPARSE[3, 2] = -243.52274
    # THETA_SPARSE[5, 2] = 773.33335
    # THETA_SPARSE[10, 2] = -974.09090
    # THETA_SPARSE[15, 2] = -0.76157
    # THETA_SPARSE[1, 3] = -773.33335
    # THETA_SPARSE[5, 3] = 2389.29628
    # THETA_SPARSE[7, 3] = -974.09090
    # THETA_SPARSE[12, 3] = -3896.36363
    # THETA_SPARSE[35, 3] = -0.76157

    ## Calculation resulting signals
    print('Calculating xLS')
    print('Calculating xSPARSE')

    LS_signals = []
    Sparse_signals = []

    for s in range(number_signals):
        def Dynamics_xLS(xLS, t):

            dxLSdt = np.zeros([dimension])

            for i in range(index_length):
                dxLSdt = dxLSdt + np.transpose(basis_functions[i](xLS)*THETA_LS[i, :])

            dxLSdt = dxLSdt + interp_inputs[s](t)

            return np.transpose(dxLSdt)


        xLS0 = np.transpose(data[s][:, 0])
        xLS = odeint(Dynamics_xLS, xLS0, tspans[s], rtol=1e-13, atol=1e-13)
        LS_signals.append(DiscreteSignal(dimension, 'LS Approximation', total_times[s], frequencies[s], signal_shape='External', data=np.transpose(xLS)))


        def Dynamics_xSPARSE(xSPARSE, t):

            dxSPARSEdt = np.zeros([dimension])

            for i in range(index_length):
                dxSPARSEdt = dxSPARSEdt + np.transpose(basis_functions[i](xSPARSE)*THETA_SPARSE[i, :])

            dxSPARSEdt = dxSPARSEdt + interp_inputs[s](t)

            return np.transpose(dxSPARSEdt)


        xSPARSE0 = np.transpose(data[s][:, 0])
        xSPARSE = odeint(Dynamics_xSPARSE, xSPARSE0, tspans[s], rtol=1e-13, atol=1e-13)
        Sparse_signals.append(DiscreteSignal(dimension, 'Sparse Approximation', total_times[s], frequencies[s], signal_shape='External', data=np.transpose(xSPARSE)))

    LS_signals_testing = []
    Sparse_signals_testing = []

    for s in range(len(x0s_testing)):
        def Dynamics_xLS(xLS, t):

            dxLSdt = np.zeros([dimension])

            for i in range(index_length):
                dxLSdt = dxLSdt + np.transpose(basis_functions[i](xLS)*THETA_LS[i, :])

            dxLSdt = dxLSdt + interp_inputs_testing[s](t)

            return np.transpose(dxLSdt)


        xLS0 = x0s_testing[s]
        xLS = odeint(Dynamics_xLS, xLS0, tspans[s], rtol=1e-13, atol=1e-13)
        LS_signals_testing.append(DiscreteSignal(dimension, 'LS Approximation test', total_times[s], frequencies[s], signal_shape='External', data=np.transpose(xLS)))


        def Dynamics_xSPARSE(xSPARSE, t):

            dxSPARSEdt = np.zeros([dimension])

            for i in range(index_length):
                dxSPARSEdt = dxSPARSEdt + np.transpose(basis_functions[i](xSPARSE)*THETA_SPARSE[i, :])

            dxSPARSEdt = dxSPARSEdt + interp_inputs_testing[s](t)

            return np.transpose(dxSPARSEdt)


        xSPARSE0 = x0s_testing[s]
        xSPARSE = odeint(Dynamics_xSPARSE, xSPARSE0, tspans[s], rtol=1e-13, atol=1e-13)
        Sparse_signals_testing.append(DiscreteSignal(dimension, 'Sparse Approximation test', total_times[s], frequencies[s], signal_shape='External', data=np.transpose(xSPARSE)))

    return interp_data, interp_inputs, index, THETA_LS, THETA_SPARSE, Y1, U, PHI, C, LS_signals, Sparse_signals, LS_signals_testing, Sparse_signals_testing
    #return interp_data, interp_inputs, index, THETA_LS, THETA_LS, Y1, U, PHI, C, LS_signals, LS_signals














































# """
# Author: Damien GUEHO
# Copyright: Copyright (C) 2021 Damien GUEHO
# License: Public Domain
# Version: 20
# Date: November 2021
# Python: 3.7.7
# """
#
#
# import numpy as np
# import cvxpy as cp
# from scipy.interpolate import interp1d
# from scipy.integrate import odeint
# import numpy.linalg as LA
#
# from SparseIDAlgorithms.GeneratePolynomialIndex import generatePolynomialIndex
# from SparseIDAlgorithms.GeneratePolynomialBasisFunctions import generatePolynomialBasisFunctions
# from ClassesGeneral.ClassSignal import DiscreteSignal, subtract2Signals
#
#
# def sparseApproximation1stOrder(signals, input_signals, order, max_order, post_treatment, l1, alpha, delta, epsilon, max_iterations, shift):
#
#     # Get signals parameters - All signals have same dimension
#     number_signals = len(signals)
#     dimension = signals[0].dimension
#     frequencies = []
#     total_times = []
#     numbers_steps = []
#     tspans = []
#     data = []
#     interp_data = []
#     inputs = []
#     interp_inputs = []
#
#     for s in range(number_signals):
#         frequencies.append(signals[s].frequency)
#         total_times.append(signals[s].total_time)
#         numbers_steps.append(signals[s].number_steps)
#         tspan = np.linspace(0, signals[s].total_time, signals[s].number_steps)
#         tspans.append(tspan)
#         data.append(signals[s].data)
#         interp_data.append(interp1d(tspan, signals[s].data, kind='previous'))
#         inputs.append(input_signals[s].data)
#         interp_inputs.append(interp1d(tspan, input_signals[s].data, kind='previous'))
#
#     for s in range(number_signals):
#         total_times[s] = total_times[s] - shift
#         numbers_steps[s] = int(numbers_steps[s] - shift * frequencies[s])
#         tspans[s] = np.linspace(0, total_times[s], numbers_steps[s])
#
#
#     # Create Index and Basis functions
#     index = generatePolynomialIndex(dimension, order, post_treatment, max_order)
#     index_length, _ = index.shape
#     basis_functions = generatePolynomialBasisFunctions(dimension, index)
#
#
#     ## Initialize Coefficient Vectors
#     THETA_LS = np.zeros([index_length, dimension])
#     THETA_SPARSE = np.zeros([index_length, dimension])
#     Y1 = np.zeros([dimension, sum(numbers_steps)])
#     U = np.zeros([dimension, sum(numbers_steps)])
#     PHI = np.zeros([sum(numbers_steps), index_length, dimension])
#     Xt = np.zeros([sum(numbers_steps), dimension])
#     C = np.zeros([dimension, index_length, max_iterations])
#
#
#     ## Integration of the N+3 equations
#     for k in range(4, dimension):
#         print('Dimension ', k + 1, ' of ', dimension)
#
#         ct = 0
#         for s in range(number_signals):
#             print('Signal number ', s + 1, ' of ', number_signals)
#
#             def Dynamics(X, t):
#
#                 dXdt = np.zeros([2 + index_length])
#
#                 x = interp_data[s](t)
#                 u = interp_inputs[s](t)
#
#                 dXdt[0] = -l1*X[0] - l1*x[k]
#                 dXdt[1] = -l1*X[1] + u[k]
#
#                 for i in range(index_length):
#                     dXdt[2 + i] = -l1*X[2 + i] + basis_functions[i](x)
#
#                 return dXdt
#
#             # Solve Differential Equation
#             y1_0 = -data[s][k, 0]
#             u_0 = 0
#             Phi_0 = np.zeros([1, index_length])
#             X0 = np.concatenate((np.array([[y1_0, u_0]]), Phi_0), axis=1)
#             X = odeint(Dynamics, X0[0, :], tspans[s], rtol=1e-13, atol=1e-13)
#             y1 = X[:, 0:1]
#             u = X[:, 1:2]
#             Phi = X[:, 2:2 + index_length]
#
#             # Define xf, xt and Phi
#             xf = np.transpose(data[s][k:k + 1, 0:numbers_steps[s]]) + y1
#             xt = xf - u
#             Y1[k:k + 1, ct:ct+numbers_steps[s]] = np.transpose(y1)
#             U[k:k + 1, ct:ct+numbers_steps[s]] = np.transpose(u)
#             PHI[ct:ct+numbers_steps[s], :, k] = Phi
#             Xt[ct:ct+numbers_steps[s], k:k+1] = xt
#             ct = ct + numbers_steps[s]
#
#
#         # Least Square Solution
#         #theta = np.matmul(LA.pinv(PHI[:, :, k]), Xt[:, k:k+1])
#         theta = np.matmul(np.matmul(LA.inv(np.matmul(PHI[:, :, k].T, PHI[:, :, k])), PHI[:, :, k].T), Xt[:, k:k+1])
#         THETA_LS[:, k:k+1] = theta
#
#         # Sparse solution
#         H = PHI[:, :, k]
#         it = 0
#
#     #     W = np.diag(np.ones(H.shape[1]))
#     #     # for i in range(H.shape[1]):
#     #     #     W[i, i] = 1 / (np.abs(theta[i, 0]) + epsilon)
#     #     # W = W / (np.max(np.abs(np.diag(W))) * 0.8 * 1e-1)
#     #
#     #     while it < max_iterations:
#     #         print('Iteration: ', it)
#     #         c = cp.Variable(shape=H.shape[1])
#     #         objective = cp.Minimize(cp.norm(W * c, 1))
#     #         constraints = [cp.norm(Xt[:, k] - H * c, 2) <= alpha * cp.norm(Xt[:, k] - np.matmul(H, theta)[:, 0], 2)]
#     #         prob = cp.Problem(objective, constraints)
#     #         prob.solve(verbose=True)
#     #         print('c', c.value)
#     #         C[k, :, it] = c.value
#     #
#     #         for i in range(H.shape[1]):
#     #             W[i, i] = 1 / (np.abs(c.value[i]) + epsilon)
#     #         W = W / (np.max(np.abs(np.diag(W))) * 0.8 * 1e-1)
#     #         #print('W', W)
#     #
#     #         it = it + 1
#     #
#     #     index_non0 = []
#     #     for i in range(H.shape[1]):
#     #         index_non0.append(i)
#     #     index_0 = []
#     #     ind = []
#     #     w = []
#     #     for i in range(H.shape[1]):
#     #         if np.abs(c.value[i]) < delta:
#     #             index_0.append(index_non0[i])
#     #             ind.append(i)
#     #         if np.abs(c.value[i]) >= delta:
#     #             w.append(c.value[i])
#     #
#     #     ind.reverse()
#     #     for i in range(len(ind)):
#     #         del index_non0[ind[i]]
#     #
#     #     print('index_non0', index_non0)
#     #     print('index_0', index_0)
#     #
#     #     H_sparse = np.take(H, index_non0, axis=1)
#     #     theta_sparse = np.matmul(LA.pinv(H_sparse), Xt[:, k:k+1])
#     #
#     #     count = 0
#     #     for i in range(index_length):
#     #         if count < len(index_non0):
#     #             if index_non0[count] == i:
#     #                 THETA_SPARSE[i, k] = theta_sparse[count, 0]
#     #                 count = count + 1
#     #
#     #
#     # ## Calculation resulting signals
#     # print('Calculating xLS')
#     # print('Calculating xSPARSE')
#
#     LS_signals = []
#     Sparse_signals = []
#
#     for s in range(number_signals):
#         def Dynamics_xLS(xLS, t):
#             print(xLS)
#
#             dxLSdt = np.zeros([dimension])
#
#             for i in range(index_length):
#                 dxLSdt = dxLSdt + np.transpose(basis_functions[i](xLS)*THETA_LS[i, :])
#
#             dxLSdt = dxLSdt + interp_inputs[s](t)
#
#             return np.transpose(dxLSdt)
#
#
#         xLS0 = np.transpose(data[s][:, 0])
#         xLS = odeint(Dynamics_xLS, xLS0, tspans[s], rtol=1e-13, atol=1e-13)
#         LS_signals.append(DiscreteSignal(dimension, 'LS Approximation', total_times[s], frequencies[s], signal_shape='External', data=np.transpose(xLS)))
#         #
#         #
#         # def Dynamics_xSPARSE(xSPARSE, t):
#         #
#         #     dxSPARSEdt = np.zeros([dimension])
#         #
#         #     for i in range(index_length):
#         #         dxSPARSEdt = dxSPARSEdt + np.transpose(basis_functions[i](xSPARSE)*THETA_SPARSE[i, :])
#         #
#         #     dxSPARSEdt = dxSPARSEdt + interp_inputs[s](t)
#         #
#         #     return np.transpose(dxSPARSEdt)
#         #
#         #
#         # xSPARSE0 = np.transpose(data[s][:, 0])
#         # xSPARSE = odeint(Dynamics_xSPARSE, xSPARSE0, tspans[s], rtol=1e-13, atol=1e-13)
#         # Sparse_signals.append(DiscreteSignal(dimension, 'Sparse Approximation', total_times[s], frequencies[s], signal_shape='External', data=np.transpose(xSPARSE)))
#
#     #return interp_data, interp_inputs, index, THETA_LS, THETA_SPARSE, Y1, U, PHI, C, LS_signals, Sparse_signals
#     return interp_data, interp_inputs, index, THETA_LS, THETA_LS, Y1, U, PHI, C, LS_signals, LS_signals
