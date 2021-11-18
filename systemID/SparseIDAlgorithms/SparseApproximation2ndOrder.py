"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


import numpy as np
from numpy import random
import cvxpy as cp
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import numpy.linalg as LA
import sys

path = sys.path[2]
sys.path.insert(1, path + '/Classes')

from SparseIDAlgorithms.GeneratePolynomialIndex import generatePolynomialIndex
from SparseIDAlgorithms.GeneratePolynomialBasisFunctions import generatePolynomialBasisFunctions
from ClassesGeneral.ClassSignal import DiscreteSignal, subtract2Signals


def sparseApproximation2ndOrder(signals, dx0s, input_signals, order, max_order, post_treatment, l1, l2, alpha, delta, epsilon, max_iterations, shift):

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
    Y2 = np.zeros([dimension, sum(numbers_steps)])
    dY2 = np.zeros([dimension, sum(numbers_steps)])
    U = np.zeros([dimension, sum(numbers_steps)])
    dU = np.zeros([dimension, sum(numbers_steps)])
    PHI = np.zeros([sum(numbers_steps), index_length, dimension])
    dPHI = np.zeros([sum(numbers_steps), index_length, dimension])
    Xt = np.zeros([sum(numbers_steps), dimension])
    C = np.zeros([dimension, index_length, max_iterations])


    ## Integration of the N+3 equations
    for k in range(dimension):
        print('Dimension ', k + 1, ' of ', dimension)

        ct = 0
        for s in range(number_signals):
            print('Signal number ', s + 1, ' of ', number_signals)

            def Dynamics(X, t):

                dXdt = np.zeros([5 + 2*index_length])

                x = interp_data[s](t)
                u = interp_inputs[s](t)

                dXdt[0] = -l2*X[0] - (l1+l2)*x[k]
                dXdt[1] = X[2]
                dXdt[2] = -(l1+l2)*X[2] - l1*l2*X[1] + l1**2*x[k]
                dXdt[3] = X[4]
                dXdt[4] = -(l1+l2)*X[4] - l1*l2*X[3] + u[k]

                dXdt[5:5 + index_length] = X[5 + index_length:5 + 2*index_length]

                for i in range(index_length):
                    dXdt[5 + index_length + i] = -(l1+l2)*X[5+index_length + i] - l1*l2*X[5 + i] + basis_functions[i](x)

                return dXdt

            # Solve Differential Equation
            y1_0 = -data[s][k, 0]
            y2_0 = 0
            dy2_0 = l1*data[s][k, 0] - dx0s[s][k, 0]
            u_0 = 0
            du_0 = 0
            Phi_0 = np.zeros([1, index_length])
            dPhi_0 = np.zeros([1, index_length])
            X0 = np.concatenate((np.array([[y1_0, y2_0, dy2_0, u_0, du_0]]), Phi_0, dPhi_0), axis=1)
            X = odeint(Dynamics, X0[0, :], tspans[s], rtol=1e-13, atol=1e-13)
            y1 = X[:, 0:1]
            y2 = X[:, 1:2]
            dy2 = X[:, 2:3]
            u = X[:, 3:4]
            du = X[:, 4:5]
            Phi = X[:, 5:5 + index_length]
            dPhi = X[:, 5 + index_length:5 + 2 * index_length]

            # Define xf, xt and Phi
            xf = np.transpose(data[s][k:k + 1, 0:numbers_steps[s]]) + y1 + y2
            xt = xf - u
            Y1[k:k + 1, ct:ct+numbers_steps[s]] = np.transpose(y1)
            Y2[k:k + 1, ct:ct+numbers_steps[s]] = np.transpose(y2)
            dY2[k:k + 1, ct:ct+numbers_steps[s]] = np.transpose(dy2)
            U[k:k + 1, ct:ct+numbers_steps[s]] = np.transpose(u)
            dU[k:k + 1, ct:ct+numbers_steps[s]] = np.transpose(du)
            PHI[ct:ct+numbers_steps[s], :, k] = Phi
            dPHI[ct:ct + numbers_steps[s], :, k] = dPhi
            Xt[ct:ct+numbers_steps[s], k:k+1] = xt
            ct = ct + numbers_steps[s]


        # Least Square Solution
        theta = np.matmul(LA.pinv(PHI[:, :, k]), Xt[:, k:k+1])
        THETA_LS[:, k:k+1] = theta

        # Sparse solution
        H = PHI[:, :, k]
        it = 0

        W = np.diag(np.ones(H.shape[1]))
        # for i in range(H.shape[1]):
        #     W[i, i] = 1 / (np.abs(theta[i, 0]) + epsilon)
        #W = W / (np.max(np.abs(np.diag(W))) * 0.8 * 1e-1)

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
            if np.abs(c.value[i]) < delta:
                index_0.append(index_non0[i])
                ind.append(i)
            if np.abs(c.value[i]) >= delta:
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


    ## Calculation resulting signals
    print('Calculating xLS')
    print('Calculating xSPARSE')

    LS_signals = []
    Sparse_signals = []

    for s in range(number_signals):
        def Dynamics_xLS(xLS, t):

            dxLSdt = np.zeros([2 * dimension])

            dxLSdt[0:dimension] = xLS[dimension:2 * dimension]

            for i in range(index_length):
                dxLSdt[dimension:2 * dimension] = dxLSdt[dimension:2 * dimension] + np.transpose(basis_functions[i](xLS[0:dimension])*THETA_LS[i, :])

            dxLSdt[dimension:2 * dimension] = dxLSdt[dimension:2 * dimension] + interp_inputs[s](t)

            return np.transpose(dxLSdt)

        xLS0 = np.transpose(data[s][:, 0])
        dxLS0 = np.transpose(dx0s[s][:, 0])
        xLS = odeint(Dynamics_xLS, np.concatenate((xLS0, dxLS0)), tspans[s], rtol=1e-13, atol=1e-13)
        LS_signals.append(DiscreteSignal(dimension, 'LS Approximation', total_times[s], frequencies[s], data=np.transpose(xLS[:, 0:dimension])))


        def Dynamics_xSPARSE(xSPARSE, t):

            dxSPARSEdt = np.zeros([2 * dimension])

            dxSPARSEdt[0:dimension] = xSPARSE[dimension:2 * dimension]

            for i in range(index_length):
                dxSPARSEdt[dimension:2 * dimension] = dxSPARSEdt[dimension:2 * dimension] + np.transpose(basis_functions[i](xSPARSE[0:dimension])*THETA_SPARSE[i, :])

            dxSPARSEdt[dimension:2 * dimension] = dxSPARSEdt[dimension:2 * dimension] + interp_inputs[s](t)

            return np.transpose(dxSPARSEdt)

        xSPARSE0 = np.transpose(data[s][:, 0])
        dxSPARSE0 = np.transpose(dx0s[s][:, 0])
        xSPARSE = odeint(Dynamics_xSPARSE, np.concatenate((xSPARSE0, dxSPARSE0)), tspans[s], rtol=1e-13, atol=1e-13)
        Sparse_signals.append(DiscreteSignal(dimension, 'Sparse Approximation', total_times[s], frequencies[s], data=np.transpose(xSPARSE[:, 0:dimension])))

    return interp_data, interp_inputs, index, THETA_LS, THETA_SPARSE, Y1, Y2, dY2, U, dU, PHI, dPHI, C, LS_signals, Sparse_signals

