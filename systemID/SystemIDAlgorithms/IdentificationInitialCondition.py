"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


import numpy as np
from scipy import linalg as LA


from systemID.SystemIDAlgorithms.GetObservabilityMatrix import getObservabilityMatrix
from systemID.SystemIDAlgorithms.GetDeltaMatrix import getDeltaMatrix

def identificationInitialCondition(input_signal, output_signal, A, B, C, D, tk, number_steps):

    # Sizes
    output_dimension, input_dimension = D(tk).shape

    # Number of steps and dt
    dt = input_signal.dt

    # Data
    u = input_signal.data[:, 0:number_steps]
    y = output_signal.data[:, 0:number_steps]

    # Build U and Y
    U = u.T.reshape(1, number_steps * input_dimension).reshape(number_steps * input_dimension, 1)
    Y = y.T.reshape(1, number_steps * output_dimension).reshape(number_steps * output_dimension, 1)

    # Get the Observability matrix
    O = getObservabilityMatrix(A, C, number_steps, tk, dt)

    # Get the Delta Matrix
    Delta = getDeltaMatrix(A, B, C, D, tk, dt, number_steps)

    # Get initial condition
    xtk1 = np.matmul(LA.pinv(O), Y - np.matmul(Delta, U))
    # print('Y', Y)
    # print('Delta', Delta)
    # print('U', U)
    # print('DeltaU', np.matmul(Delta, U))
    # print('Y - DeltaU', Y - np.matmul(Delta, U))
    # print('O', O)
    # u, s, v = LA.svd(O)
    # print('s', s)
    # # # print('Shape Y:', Y.shape)
    # # print('LA.pinv(O)', LA.pinv(O))
    # # print('O*LA.pinv(O)', np.matmul(O, LA.pinv(O)))
    # print('LA.pinv(O)*O', np.matmul(LA.pinv(O), O))
    # print('OtO-1Ot', np.matmul(LA.inv(np.matmul(O.T, O)), O.T))
    # print('O*OtO-1Ot', np.matmul(O, np.matmul(LA.inv(np.matmul(O.T, O)), O.T)))
    # print('OtO-1Ot*O', np.matmul(np.matmul(LA.inv(np.matmul(O.T, O)), O.T), O))
    # # # print('Diff pinv', LA.pinv(O) - np.matmul(LA.inv(np.matmul(O.T, O)), O.T))
    # # print('--------------------------------------------------------------------')
    print('Error IC pinv: ', LA.norm(Y - np.matmul(O, xtk1) - np.matmul(Delta, U)))
    # print('Diff: ', Y - np.matmul(O, xtk1) - np.matmul(Delta, U))
    # print('--------------------------------------------------------------------')
    # xtk2 = np.matmul(np.matmul(LA.inv(np.matmul(O.T, O)), O.T), Y - np.matmul(Delta, U))
    # print('Error IC classical: ', LA.norm(Y - np.matmul(O, xtk2) - np.matmul(Delta, U)))
    # print('Diff: ', Y - np.matmul(O, xtk2) - np.matmul(Delta, U))

    return xtk1[:, 0]

