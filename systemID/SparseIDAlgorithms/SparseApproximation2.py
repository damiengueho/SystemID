"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""


import numpy as np
from numpy import random
import cvxpy as cp
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import numpy.linalg as LA

from GenerateIndex import generateIndex
from GenerateBasisFunctions import generateBasisFunctions


def sparseApproximation2(signal, dx0, input_signal, order, max_order, post_treatment, l1, l2, alpha, delta, max_iterations):

    # Dimension and frequency
    frequency = signal.frequency
    dimension = signal.dimension

    # Time span
    total_time = signal.total_time
    number_steps = signal.number_steps
    tspan = np.linspace(0, total_time, number_steps)

    # Signal to be considered
    x = signal.data
    #x = x/np.max(np.abs(x))
    interp_x = interp1d(tspan, x, kind='cubic')

    # Input Signal
    u = input_signal.data
    interp_u = interp1d(tspan, u, kind='cubic')

    # New total_time and number_Steps
    total_time = total_time - 1
    number_steps = number_steps - 1 * frequency
    tspan = np.linspace(0, total_time, number_steps)

    # Create Index and Basis functions
    index = generateIndex(dimension, order, post_treatment, max_order)
    print('Index: ', index)
    index_length, _ = index.shape
    basis_functions = generateBasisFunctions(dimension, index)

    # Initialize Coefficient Vectors
    THETA_LS = np.zeros([index_length, dimension])
    THETA_SPARSE = np.zeros([index_length, dimension])
    ZX = np.zeros([dimension, number_steps])
    ZV = np.zeros([dimension, number_steps])
    ZL = np.zeros([dimension, number_steps])
    ZL_dot = np.zeros([dimension, number_steps])
    ZU = np.zeros([dimension, number_steps])
    ZU_dot = np.zeros([dimension, number_steps])

    for k in range(dimension):
        print('Dimension ', k + 1, ' of ', dimension)

        # Create Dynamics for zx, zu and Psix for each dimension
        def Dynamics(X, t):
            dXdt = np.zeros([6 + 2 * index_length])

            zx = X[0]
            zv = X[1]
            zl = X[2]
            zl_dot = X[3]
            zu = X[4]
            zu_dot = X[5]

            dXdt[0] = -l2 * zx + interp_x(t)[k]
            dXdt[1] = -l1 * zv + interp_x(t)[k]
            dXdt[2] = zl_dot
            dXdt[3] = - (l1 + l2) * zl_dot - l1 * l2 * zl + interp_x(t)[k]
            dXdt[4] = zu_dot
            dXdt[5] = - (l1 + l2) * zu_dot - l1 * l2 * zu + interp_u(t)[k]

            dXdt[6:6 + index_length] = X[6 + index_length:6 + 2 * index_length]

            for i in range(index_length):
                Psi = X[6 + i]
                Psi_dot = X[6 + index_length + i]
                dXdt[6 + i] = Psi_dot
                dXdt[6 + index_length + i] = - (l1 + l2) * Psi_dot - l1 * l2 * Psi + basis_functions[i](interp_x(t))

            return dXdt

        ## Create data Set for zx, zu and Psix - Solve Differential Equation
        zx0 = x[k, 0] / l2
        zv0 = 0
        zl0 = 0
        dzl0 = x[k, 0] / l2 - dx0[k, 0] / (l1 * l2)
        zu0 = 0
        dzu0 = 0
        Psi0 = np.zeros([1, index_length])
        dPsi0 = np.zeros([1, index_length])
        X0 = np.concatenate((np.array([[zx0, zv0, zl0, dzl0, zu0, dzu0]]), Psi0, dPsi0), axis=1)
        X = odeint(Dynamics, X0[0, :], tspan, rtol=1e-13, atol=1e-13)
        zx = X[:, 0:1]
        zv = X[:, 1:2]
        zl = X[:, 2:3]
        zl_dot = X[:, 3:4]
        zu = X[:, 4:5]
        zu_dot = X[:, 5:6]
        Psi = X[:, 6:6 + index_length]
        Psi_dot = X[:, 6 + index_length:6 + 2 * index_length]

        ## Define xf and xf_new
        xf = np.transpose(x[k:k + 1, 0:number_steps]) - l2 * zx - l1 * zv + l1 * l2 * zl
        y = xf - zu

        ## Least Square solution
        H = np.zeros([number_steps, index_length])
        H[:, 0:index_length] = Psi
        theta = np.matmul(LA.pinv(H), y)

        # yID = np.matmul(H, theta)[:, 0:dimension]
        # xfID = yID

        THETA_LS[:, k:k + 1] = theta
        print('theta', theta)
        ZX[k:k + 1, :] = np.transpose(zx)
        ZV[k:k + 1, :] = np.transpose(zv)
        ZL[k:k + 1, :] = np.transpose(zl)
        ZL_dot[k:k + 1, :] = np.transpose(zl_dot)
        ZU[k:k + 1, :] = np.transpose(zu)
        ZU_dot[k:k + 1, :] = np.transpose(zu_dot)

        ## Sparse solution
        #theta = np.random.rand(index_length, 1)
        #theta = np.zeros([index_length, 1])
        it = 0
        index_non0 = []
        for i in range(H.shape[1]):
            index_non0.append(i)
        index_0 = []
        H_initial = H
        W = np.diag(np.ones(H.shape[1]))
        for i in range(H.shape[1]):
            W[i, i] = 1 / (np.abs(theta[i, 0]) + delta)
        W = W / (np.max(np.abs(np.diag(W))) * 0.8 * 1e-1)

        while it < max_iterations:
            print('Iteration: ', it)
            c = cp.Variable(shape=H.shape[1])
            objective = cp.Minimize(cp.norm(W * c, 1))
            constraints = [cp.norm(y[:, 0] - H * c, 2) <= alpha * cp.norm(y[:, 0] - np.matmul(H, theta)[:, 0], 2)]
            prob = cp.Problem(objective, constraints)
            prob.solve()
            print('c', c.value)

            e_norm = np.mean((y[:, 0] - H * c) ** 2)

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

            H_sparse = np.take(H_initial, index_non0, axis=1)
            theta_sparse = np.matmul(LA.pinv(H_sparse), y)

            print('theta_sparse', theta_sparse)

            # check = not(theta.shape==theta_sparse.shape)
            theta = theta_sparse
            H = H_sparse
            it = it + 1

            W = np.diag(np.ones(H.shape[1]))
            for i in range(H.shape[1]):
                W[i, i] = 1 / (np.abs(w[i]) + delta)

            W = W / (np.max(np.abs(np.diag(W))) * 0.8 * 1e-1)
            print('W', W)

        count = 0
        for i in range(index_length):
            if count < len(index_non0):
                if index_non0[count] == i:
                    THETA_SPARSE[i, k] = theta_sparse[count, 0]
                    count = count + 1


    print('Calculating xLS')

    def Dynamics_xLS(xLS, t):

        dxLSdt = np.zeros([2 * dimension])

        dxLSdt[0:dimension] = xLS[dimension:2 * dimension]

        for i in range(index_length):
            dxLSdt[dimension:2 * dimension] = dxLSdt[dimension:2 * dimension] + np.transpose(basis_functions[i](xLS[0:dimension]) * THETA_LS[i, :])

        dxLSdt[dimension:2 * dimension] = dxLSdt[dimension:2 * dimension] + interp_u(t)

        return np.transpose(dxLSdt)

    xLS0 = np.transpose(x[:, 0])
    dxLS0 = np.transpose(dx0[:, 0])
    xLS = odeint(Dynamics_xLS, np.concatenate((xLS0, dxLS0)), tspan, rtol=1e-13, atol=1e-13)


    print('Calculating xSPARSE')


    def Dynamics_xSPARSE(xSPARSE, t):

        dxSPARSEdt = np.zeros([2 * dimension])

        dxSPARSEdt[0:dimension] = xSPARSE[dimension:2 * dimension]

        for i in range(index_length):
            dxSPARSEdt[dimension:2 * dimension] = dxSPARSEdt[dimension:2 * dimension] + np.transpose(basis_functions[i](xSPARSE[0:dimension]) * THETA_SPARSE[i, :])

        dxSPARSEdt[dimension:2 * dimension] = dxSPARSEdt[dimension:2 * dimension] + interp_u(t)

        return np.transpose(dxSPARSEdt)

    xSPARSE0 = np.transpose(x[:, 0])
    dxSPARSE0 = np.transpose(dx0[:, 0])
    xSPARSE = odeint(Dynamics_xSPARSE, np.concatenate((xSPARSE0, dxSPARSE0)), tspan, rtol=1e-13, atol=1e-13)


    return(interp_x, interp_u, THETA_LS, THETA_SPARSE, ZX, ZV, ZL, ZL_dot, ZU, ZU_dot, Psi, Psi_dot, np.transpose(xLS), np.transpose(xSPARSE))

