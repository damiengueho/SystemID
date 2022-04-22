"""
Author: Damien GUEHO
Copyright: Copyright (C) 2022 Damien GUEHO
License: Public Domain
Version: 23
Date: April 2022
Python: 3.7.7
"""


import numpy as np
import scipy.linalg as LA

from systemID.ClassesSystemID.ClassERA import ERAFromInitialConditionResponse
from systemID.ClassesGeneral.ClassSignal import ContinuousSignal, OutputSignal, DiscreteSignal


def bilinearSystemID(experiments_1, experiments_2, state_dimension, dt, **kwargs):

    ## Dimensions
    output_dimension = experiments_1.output_dimension
    input_dimension = experiments_1.input_dimension
    N1 = experiments_1.number_experiments
    N2 = len(experiments_2)
    p = kwargs.get('p', )


    ## Identification of D
    Y0_1N1 = np.zeros([output_dimension, N1])
    V0_1N1 = np.zeros([input_dimension, N1])

    for i in range(N1):
        Y0_1N1[:, i] = experiments_1.output_signals[i].data[:, 0]
        V0_1N1[:, i] = experiments_1.input_signals[i].data[:, 0]
    D = np.matmul(Y0_1N1, LA.pinv(V0_1N1))


    ## Identification of C and A
    Y1_pN1 = np.zeros([p * output_dimension, N1])
    Y2_pN1 = np.zeros([p * output_dimension, N1])
    Y1_pN2 = np.zeros([p * output_dimension, N2])

    for i in range(p):
        for j in range(N1):
            Y1_pN1[i * output_dimension:(i + 1) * output_dimension, j] = experiments_1.output_signals[j].data[:, i + 1]
        for j in range(N1):
            Y2_pN1[i * output_dimension:(i + 1) * output_dimension, j] = experiments_1.output_signals[j].data[:, i + 2]

    (R1, sigma1, St1) = LA.svd(Y1_pN1)
    Sigma1 = np.diag(sigma1)
    Rn1 = R1[:, 0:state_dimension]
    Snt1 = St1[0:state_dimension, :]
    Sigman1 = Sigma1[0:state_dimension, 0:state_dimension]
    O1 = np.matmul(Rn1, LA.sqrtm(Sigman1))
    X1 = np.matmul(LA.sqrtm(Sigman1), Snt1)

    C = O1[0:output_dimension, :]
    AA = np.matmul(LA.pinv(O1), np.matmul(Y2_pN1, LA.pinv(X1)))
    print(LA.eig(AA)[0])
    Ac = LA.logm(AA) / dt


    ## Identification of Nc
    LAA = np.zeros([state_dimension, N2 * state_dimension])

    for i in range(N2):
        Y2_pN2 = np.zeros([p * output_dimension, N2])
        for j in range(N2):
            for k in range(p):
                Y2_pN2[k * output_dimension:(k + 1) * output_dimension, j] = experiments_2[i].output_signals[j].data[:, k + 2]

        P = Y2_pN2 - np.matmul(O1, np.outer(X1[:, i:i + 1], np.ones(N2)))

        LAA[:, i * state_dimension:(i + 1) * state_dimension] = LA.logm(np.matmul(LA.pinv(O1), np.matmul(P, LA.pinv(X1[:, 0:N2])))) / dt - Ac

    CR = np.kron(V0_1N1[:, 0:N2], np.eye(state_dimension))
    CR_inv = LA.pinv(CR)
    Nc = np.matmul(LAA, CR_inv)


    ## Identification of Bc
    Z = np.zeros([2 * state_dimension, 2 * state_dimension])
    Coeff = np.zeros([state_dimension * N1, input_dimension * state_dimension])
    ytil = np.zeros([state_dimension * N1])

    for i in range(N1):
        Z[0:state_dimension, 0:state_dimension] = Ac
        for j in range(input_dimension):
            Z[0:state_dimension, 0:state_dimension] += Nc[:, j * state_dimension:(j + 1) * state_dimension] * V0_1N1[j, i]
        Z[0:state_dimension, state_dimension:2 * state_dimension] = np.eye(state_dimension)
        expZ = LA.expm(Z * dt)
        G = expZ[0:state_dimension, state_dimension:2 * state_dimension]
        Coeff[i * state_dimension:(i + 1) * state_dimension, :] = np.kron(V0_1N1[:, i:i + 1].T, G)
        ytil[i * state_dimension:(i + 1) * state_dimension] = X1[:, i]

    VecBchat = np.matmul(LA.pinv(Coeff), ytil)
    Bc = np.reshape(VecBchat, [input_dimension, state_dimension]).T


    ## Identified system matrices
    def A_id(t):
        return Ac
    def N_id(t):
        return Nc
    def B_id(t):
        return Bc
    def C_id(t):
        return C
    def D_id(t):
        return D

    return A_id, N_id, B_id, C_id, D_id, Sigma1









def bilinearSystemIDFromInitialConditionResponse(experiments_1, experiments_2, state_dimension, dt, **kwargs):

    ## Dimensions
    output_dimension = experiments_1.output_dimension
    input_dimension = experiments_1.input_dimension
    N1 = experiments_1.number_experiments
    N2 = len(experiments_2)
    p = kwargs.get('p', )


    ## ERAIC for C, A0 and A
    eraic = ERAFromInitialConditionResponse(experiments_1.output_signals, state_dimension, input_dimension, p=p, q=p)
    Op = eraic.Op
    X0 = eraic.X0

    C = eraic.C(0)
    Ac = LA.logm(eraic.A(0)) / dt


    ## Identification of Nc
    LAA = np.zeros([state_dimension, N2 * state_dimension])

    for i in range(N2):
        Y1_pN2 = np.zeros([p * output_dimension, N2])
        for j in range(N2):
            for k in range(p):
                Y1_pN2[k * output_dimension:(k + 1) * output_dimension, j] = experiments_2[i].output_signals[j].data[:, k + 1]

        LAA[:, i * state_dimension:(i + 1) * state_dimension] = LA.logm(np.matmul(LA.pinv(Op), np.matmul(Y1_pN2, LA.pinv(X0[:, 0:N2])))) / dt - Ac

    V0_1N2 = np.zeros([input_dimension, N1])
    for i in range(N1):
        V0_1N2[:, i] = experiments_2[i].input_signals[0].data[:, 0]

    CR = np.kron(V0_1N2[:, 0:N2], np.eye(state_dimension))
    CR_inv = LA.pinv(CR)
    Nc = np.matmul(LAA, CR_inv)


    ## Identified system matrices
    def A_id(t):
        return Ac
    def N_id(t):
        return Nc
    def B_id(t):
        return np.zeros([state_dimension, input_dimension])
    def C_id(t):
        return C
    def D_id(t):
        return np.zeros([output_dimension, input_dimension])

    return A_id, N_id, B_id, C_id, D_id, X0, eraic.Sigma










def bilinearSystemIDFromInitialConditionResponseConstantInput(experiments_1, experiments_2, state_dimension, dt, ind, l, **kwargs):

    ## Dimensions
    output_dimension = experiments_1.output_dimension
    input_dimension = experiments_1.input_dimension
    frequency = experiments_1.frequency
    N1 = experiments_1.number_experiments
    N2 = len(experiments_2)
    p = kwargs.get('p', )


    ## Extracting relevant training data experiments_1
    inputs_1 = []
    outputs_1 = []
    for i in range(N1):
        inputs_1.append(DiscreteSignal(input_dimension, l * dt, frequency))
        output_signal = experiments_1.output_signals[i]
        outputs_1.append(DiscreteSignal(output_dimension, l * dt, frequency, signal_shape='External', data=output_signal.data[:, ind:ind + l + 1]))
    experiments_1.input_signals = inputs_1
    experiments_1.output_signals = outputs_1
    experiments_1.tspan = np.linspace(0, l * dt, l + 1)
    experiments_1.total_time = l * dt

    ## Extracting relevant training data experiments_2
    for i in range(N2):
        experiment_2 = experiments_2[i]
        inputs_2 = []
        outputs_2 = []
        for j in range(N2):
            input_signal = experiment_2.input_signals[j]
            inputs_2.append(DiscreteSignal(input_dimension, l * dt, frequency, signal_shape='External', data=input_signal.data[:, ind:ind + l + 1]))
            output_signal = experiment_2.output_signals[j]
            outputs_2.append(DiscreteSignal(output_dimension, l * dt, frequency, signal_shape='External', data=output_signal.data[:, ind:ind + l + 1]))
        experiments_2[i].input_signals = inputs_2
        experiments_2[i].output_signals = outputs_2
        experiments_2[i].tspan = np.linspace(0, l * dt, l + 1)
        experiments_2[i].total_time = l * dt

    ## ERAIC for C, A0 and A
    eraic = ERAFromInitialConditionResponse(experiments_1.output_signals, state_dimension, input_dimension, p=p, q=p)
    Op = eraic.Op
    X0 = eraic.X0

    C = eraic.C(0)
    Ac = LA.logm(eraic.A(0)) / dt

    ## Identification of Nc
    LAA = np.zeros([state_dimension, N2 * state_dimension])

    for i in range(N2):
        Y1_pN2 = np.zeros([p * output_dimension, N2])
        y0_1N2 = np.zeros([output_dimension, N2])
        for j in range(N2):
            for k in range(p):
                Y1_pN2[k * output_dimension:(k + 1) * output_dimension, j] = experiments_2[i].output_signals[j].data[:, k + 1]

            y0_1N2[:, j] = experiments_2[i].output_signals[j].data[:, 0]
            
        xp = np.matmul(LA.pinv(C), y0_1N2)

        LAA[:, i * state_dimension:(i + 1) * state_dimension] = LA.logm(np.matmul(LA.pinv(Op), np.matmul(Y1_pN2, LA.pinv(xp)))) / dt - Ac

    V0_1N2 = np.zeros([input_dimension, N1])
    for i in range(N1):
        V0_1N2[:, i] = experiments_2[i].input_signals[0].data[:, 0]

    CR = np.kron(V0_1N2[:, 0:N2], np.eye(state_dimension))
    CR_inv = LA.pinv(CR)
    Nc = np.matmul(LAA, CR_inv)

    ## Identified system matrices
    def A_id(t):
        return Ac

    def N_id(t):
        return Nc

    def B_id(t):
        return np.zeros([state_dimension, input_dimension])

    def C_id(t):
        return C

    def D_id(t):
        return np.zeros([output_dimension, input_dimension])

    return A_id, N_id, B_id, C_id, D_id, X0, eraic.Sigma, LAA, V0_1N2, CR, Op, experiments_1, experiments_2









def bilinearSystemIDFromInitialConditionResponseConstantInput2(experiments, state_dimension, dt, **kwargs):

    ## Dimensions
    output_dimension = experiments.output_dimension
    input_dimension = experiments.input_dimension
    N = experiments.number_experiments
    p = kwargs.get('p', )


    ## ERAIC for C, A0 and A
    eraic = ERAFromInitialConditionResponse([experiments.output_signals[0]], state_dimension, input_dimension, p=p, q=p)
    Op = eraic.Op
    X0 = eraic.X0
    print(eraic.X0)

    C = eraic.C(0)
    Ac = LA.logm(eraic.A(0)) / dt


    ## Identification of Nc
    N2 = N - 1
    LAA = np.zeros([state_dimension, N2 * state_dimension])

    for i in range(N2):
        eraic = ERAFromInitialConditionResponse([experiments.output_signals[i + 1]], state_dimension, input_dimension, p=p, q=p)
        Aj = np.matmul(np.matmul(LA.pinv(C), eraic.C(0)), eraic.A(0))
        print(np.matmul(np.matmul(LA.pinv(C), eraic.C(0)), eraic.X0))

        LAA[:, i * state_dimension:(i + 1) * state_dimension] = LA.logm(Aj) / dt - Ac

    V0_1N2 = np.zeros([input_dimension, N2])
    for i in range(N2):
        V0_1N2[:, i] = experiments.input_signals[i + 1].data[:, 0]

    CR = np.kron(V0_1N2[:, 0:N2], np.eye(state_dimension))
    CR_inv = LA.pinv(CR)
    Nc = np.matmul(LAA, CR_inv)


    ## Identified system matrices
    def A_id(t):
        return Ac
    def N_id(t):
        return Nc
    def B_id(t):
        return np.zeros([state_dimension, input_dimension])
    def C_id(t):
        return C
    def D_id(t):
        return np.zeros([output_dimension, input_dimension])

    return A_id, N_id, B_id, C_id, D_id, X0, eraic.Sigma, LAA, V0_1N2, CR, Op