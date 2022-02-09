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
import sys

path = sys.path[2]
sys.path.insert(1, path + '/Classes')

from GenerateIndex import generateIndex
from GenerateBasisFunctions import generateBasisFunctions
from ClassSignal import Signal, subtract2Signals


def sparseApproximation2_TwoBodyProblem(signals, dx0s, input_signals, order, max_order, post_treatment, l1, l2, alpha, delta, max_iterations, TU, shift):

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
        tspan = np.linspace(0, signals[s].total_time, signals[s].number_steps) / TU
        tspans.append(tspan)
        data.append(signals[s].data)
        interp_data.append(interp1d(tspan, signals[s].data, kind='cubic'))
        inputs.append(input_signals[s].data)
        interp_inputs.append(interp1d(tspan, input_signals[s].data, kind='cubic'))


    for s in range(number_signals):
        total_times[s] = total_times[s] - shift
        numbers_steps[s] = int(numbers_steps[s] - shift * frequencies[s])
        tspans[s] = np.linspace(0, total_times[s], numbers_steps[s]) / TU


    # Create Index and Basis functions
    index0 = generateIndex(dimension + 1, order, post_treatment, max_order)
    index0_length, _ = index0.shape
    basis_functions_temp = generateBasisFunctions(dimension + 1, index0)
    rep = []

    basis_functions = []
    index = np.zeros([index0_length * 5, 2 * (dimension + 1)])
    c = 0
    for i in range(index0_length):
        if int(index0[i][3]) == 0:
            basis_functions.append(basis_functions_temp[i])
            index[c, 0:dimension + 1] = index0[i]
            c+=1

    for i in range(index0_length):
        for j in range(index0_length):
            #np.sum(index0[i]) < np.sum(index0[j])
            if int(index0[i][3]) == 0 and int(index0[j][0]) == 0 and int(index0[j][1]) == 0 and int(index0[j][2]) == 0 and int(index0[j][3]) > 0:
                def make_Phix(i, j):
                    def Phix(x):
                        return basis_functions_temp[i](x) / basis_functions_temp[j](x)
                    return Phix
                basis_functions.append(make_Phix(i, j))
                index[c, 0:dimension + 1] = index0[i]
                index[c, dimension + 1:2 * (dimension + 1)] = index0[j]
                c+=1
    index_length = len(basis_functions)
    print(index_length)


    # def Phix(x):
    #     return 1
    # basis_functions = [Phix]
    # index = np.zeros([index0_length * index0_length, 2 * (dimension + 1)])
    # c = 1
    # for i in range(index0_length):
    #     for j in range(index0_length):
    #         possible_rep = (index0[i] - index0[j]).tolist()
    #         if np.sum(index0[i]) < np.sum(index0[j]) and not(possible_rep in rep):
    #             rep.append(possible_rep)
    #             if int(index0[i][3]) > 0 or int(index0[j][3]) > 0:
    #                 def make_Phix(i, j):
    #                     def Phix(x):
    #                         return basis_functions_temp[i](x) / basis_functions_temp[j](x)
    #                     return Phix
    #                 basis_functions.append(make_Phix(i, j))
    #                 index[c, 0:dimension + 1] = index0[i]
    #                 index[c, dimension + 1:2 * (dimension + 1)] = index0[j]
    #                 c+=1
    # index_length = len(basis_functions)
    # print(index_length)
    #
    # index_length = 9
    #
    # def Phi1(x):
    #     r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    #     return x[0] / (r ** 3)
    # def Phi2(x):
    #     r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    #     return x[1] / (r ** 3)
    # def Phi3(x):
    #     r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    #     return x[2] / (r ** 3)
    # def Phi4(x):
    #     return x[0]*x[1] / (np.sqrt(x[0]**2+x[1]**2+x[2]**2) ** 3)
    # def Phi5(x):
    #     return x[1]*x[2] / (np.sqrt(x[0]**2+x[1]**2+x[2]**2) ** 3)
    # def Phi6(x):
    #     return x[2]*x[0] / (np.sqrt(x[0]**2+x[1]**2+x[2]**2) ** 3)
    # def Phi7(x):
    #     return x[0] ** 2 / (np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 3)
    #
    # def Phi8(x):
    #     return x[1] ** 2 / (np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 3)
    #
    # def Phi9(x):
    #     return x[2] ** 2 / (np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 3)
    #
    #
    # basis_functions = [Phi1, Phi2, Phi3, Phi4, Phi5, Phi6, Phi7, Phi8, Phi9]

    # Initialize Coefficient Vectors
    THETA_LS = np.zeros([index_length, dimension])
    THETA_SPARSE = np.zeros([index_length, dimension])
    ZX = np.zeros([dimension, sum(numbers_steps)])
    ZV = np.zeros([dimension, sum(numbers_steps)])
    ZL = np.zeros([dimension, sum(numbers_steps)])
    ZL_dot = np.zeros([dimension, sum(numbers_steps)])
    ZU = np.zeros([dimension, sum(numbers_steps)])
    ZU_dot = np.zeros([dimension, sum(numbers_steps)])
    PSI = np.zeros([sum(numbers_steps), index_length, dimension])
    Y = np.zeros([sum(numbers_steps), dimension])

    for k in range(dimension):
        print('Dimension ', k + 1, ' of ', dimension)

        ct = 0
        for s in range(number_signals):
            def Dynamics(X, t):
                dXdt = np.zeros([6 + 2 * index_length])

                zx = X[0]
                zv = X[1]
                zl = X[2]
                zl_dot = X[3]
                zu = X[4]
                zu_dot = X[5]

                dXdt[0] = -l2 * zx + interp_data[s](t)[k]
                dXdt[1] = -l1 * zv + interp_data[s](t)[k]
                dXdt[2] = zl_dot
                dXdt[3] = - (l1 + l2) * zl_dot - l1 * l2 * zl + interp_data[s](t)[k]
                dXdt[4] = zu_dot
                dXdt[5] = - (l1 + l2) * zu_dot - l1 * l2 * zu + interp_inputs[s](t)[k]

                dXdt[6:6 + index_length] = X[6 + index_length:6 + 2 * index_length]

                r = np.array([LA.norm(interp_data[s](t))])

                for i in range(index_length):
                    Psi = X[6 + i]
                    Psi_dot = X[6 + index_length + i]
                    dXdt[6 + i] = Psi_dot
                    dXdt[6 + index_length + i] = - (l1 + l2) * Psi_dot - l1 * l2 * Psi + basis_functions[i](np.concatenate((interp_data[s](t), r)))

                return dXdt

            ## Create data Set for zx, zu and Psix - Solve Differential Equation
            zx0 = data[s][k, 0] / l2
            zv0 = 0
            zl0 = 0
            dzl0 = data[s][k, 0] / l2 - dx0s[s][k, 0] / (l1 * l2)
            zu0 = 0
            dzu0 = 0
            Psi0 = np.zeros([1, index_length])
            dPsi0 = np.zeros([1, index_length])
            X0 = np.concatenate((np.array([[zx0, zv0, zl0, dzl0, zu0, dzu0]]), Psi0, dPsi0), axis=1)
            X = odeint(Dynamics, X0[0, :], tspans[s], rtol=1e-13, atol=1e-13)
            zx = X[:, 0:1]
            zv = X[:, 1:2]
            zl = X[:, 2:3]
            zl_dot = X[:, 3:4]
            zu = X[:, 4:5]
            zu_dot = X[:, 5:6]
            Psi = X[:, 6:6 + index_length]
            Psi_dot = X[:, 6 + index_length:6 + 2 * index_length]

            ## Define xf and y
            xf = np.transpose(data[s][k:k + 1, 0:numbers_steps[s]]) - l2 * zx - l1 * zv + l1 * l2 * zl
            y = xf - zu
            ZX[k:k + 1, ct:ct+numbers_steps[s]] = np.transpose(zx)
            ZV[k:k + 1, ct:ct+numbers_steps[s]] = np.transpose(zv)
            ZL[k:k + 1, ct:ct+numbers_steps[s]] = np.transpose(zl)
            ZL_dot[k:k + 1, ct:ct+numbers_steps[s]] = np.transpose(zl_dot)
            ZU[k:k + 1, ct:ct+numbers_steps[s]] = np.transpose(zu)
            ZU_dot[k:k + 1, ct:ct+numbers_steps[s]] = np.transpose(zu_dot)
            PSI[ct:ct+numbers_steps[s], :, k] = Psi
            Y[ct:ct+numbers_steps[s], k:k+1] = y
            ct = ct + numbers_steps[s]


        # Least Square Solution
        theta = np.matmul(LA.pinv(PSI[:, :, k]), Y[:, k:k+1])
        THETA_LS[:, k:k+1] = theta



        ## Sparse solution
        #theta = np.random.rand(index_length, 1)
        #theta = np.zeros([index_length, 1])
        H = PSI[:, :, k]
        it = 0
        index_non0 = []
        for i in range(H.shape[1]):
            index_non0.append(i)
        index_0 = []
        H_initial = H
        W = np.diag(np.ones(H.shape[1]))
        # for i in range(H.shape[1]):
        #     W[i, i] = 1 / (np.abs(theta[i, 0]) + delta)
        # W = W / (np.max(np.abs(np.diag(W))) * 0.8 * 1e-1)

        while it < max_iterations:
            print('Iteration: ', it)
            c = cp.Variable(shape=H.shape[1])
            objective = cp.Minimize(cp.norm(W * c, 1))
            constraints = [cp.norm(Y[:, k] - H * c, 2) <= alpha * cp.norm(Y[:, k] - np.matmul(H, theta)[:, 0], 2)]
            prob = cp.Problem(objective, constraints)
            prob.solve()
            print('c', c.value)

            e_norm = np.mean((Y[:, k] - H * c) ** 2)
            print(e_norm.value)

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
            theta_sparse = np.matmul(LA.pinv(H_sparse), Y[:, k:k + 1])

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

        #     e_norm = np.mean((Y[:, k] - H * c) ** 2)
        #     print(e_norm.value)
        #
        #     ind = []
        #     w = []
        #     for i in range(H.shape[1]):
        #         if np.abs(c.value[i]) < delta:
        #             index_0.append(index_non0[i])
        #             ind.append(i)
        #         if np.abs(c.value[i]) >= delta:
        #             w.append(c.value[i])
        #
        #     ind.reverse()
        #     for i in range(len(ind)):
        #         del index_non0[ind[i]]
        #
        #     print('index_non0', index_non0)
        #     print('index_0', index_0)
        #
        #     H_sparse = np.take(H_initial, index_non0, axis=1)
        #     theta_sparse = np.matmul(LA.pinv(H_sparse), Y[:, k:k+1])
        #
        #     print('theta_sparse', theta_sparse)
        #
        #     # check = not(theta.shape==theta_sparse.shape)
        #     theta = theta_sparse
        #     H = H_sparse
        #     it = it + 1
        #
        #     W = np.diag(np.ones(H.shape[1]))
        #     for i in range(H.shape[1]):
        #         W[i, i] = 1 / (np.abs(w[i]) + delta)
        #
        #     W = W / (np.max(np.abs(np.diag(W))) * 0.8 * 1e-1)
        #     print('W', W)
        #
        # count = 0
        # for i in range(index_length):
        #     if count < len(index_non0):
        #         if index_non0[count] == i:
        #             THETA_SPARSE[i, k] = theta_sparse[count, 0]
        #             count = count + 1


    print('Calculating xLS')
    print('Calculating xSPARSE')



    LS_signals = []
    Sparse_signals = []
    for s in range(number_signals):
        def Dynamics_xLS(xLS, t):

            dxLSdt = np.zeros([2 * dimension])

            dxLSdt[0:dimension] = xLS[dimension:2 * dimension]

            r = np.array([LA.norm(xLS[0:dimension])])

            for i in range(index_length):
                dxLSdt[dimension:2 * dimension] = dxLSdt[dimension:2 * dimension] + np.transpose(basis_functions[i](np.concatenate((xLS[0:dimension], r))) * THETA_LS[i, :])

            dxLSdt[dimension:2 * dimension] = dxLSdt[dimension:2 * dimension] + interp_inputs[s](t)

            return np.transpose(dxLSdt)

        xLS0 = np.transpose(data[s][:, 0])
        dxLS0 = np.transpose(dx0s[s][:, 0])
        xLS = odeint(Dynamics_xLS, np.concatenate((xLS0, dxLS0)), tspans[s], rtol=1e-13, atol=1e-13)
        LS_signals.append(Signal(total_times[s], frequencies[s], dimension, 'LS Approximation', data=np.transpose(xLS[:, 0:dimension])))


        def Dynamics_xSPARSE(xSPARSE, t):

            dxSPARSEdt = np.zeros([2 * dimension])

            dxSPARSEdt[0:dimension] = xSPARSE[dimension:2 * dimension]

            r = np.array([LA.norm(xSPARSE[0:dimension])])

            for i in range(index_length):
                dxSPARSEdt[dimension:2 * dimension] = dxSPARSEdt[dimension:2 * dimension] + np.transpose(basis_functions[i](np.concatenate((xSPARSE[0:dimension], r))) * THETA_SPARSE[i, :])

            dxSPARSEdt[dimension:2 * dimension] = dxSPARSEdt[dimension:2 * dimension] + interp_inputs[s](t)

            return np.transpose(dxSPARSEdt)

        xSPARSE0 = np.transpose(data[s][:, 0])
        dxSPARSE0 = np.transpose(dx0s[s][:, 0])
        xSPARSE = odeint(Dynamics_xSPARSE, np.concatenate((xSPARSE0, dxSPARSE0)), tspans[s], rtol=1e-13, atol=1e-13)
        Sparse_signals.append(Signal(total_times[s], frequencies[s], dimension, 'Sparse Approximation', data=np.transpose(xSPARSE[:, 0:dimension])))

    return(interp_data, interp_inputs, index, THETA_LS, THETA_SPARSE, ZX, ZV, ZL, ZL_dot, ZU, ZU_dot, PSI, LS_signals, Sparse_signals)

