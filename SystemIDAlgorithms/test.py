"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 17
Date: October 2021
Python: 3.7.7
"""


import numpy as np
from sympy import *
x, y, z, t, u = symbols('x y z t u')

dimension = 2
def delta(t):
    return 2
def alpha(t):
    return 3
def beta(t):
    return 4

def F(x, t, u):
    dxdt = np.zeros(dimension)
    dxdt[0] = x[1]
    dxdt[1] = -delta(t) * x[1] - alpha(t) * x[0] - beta(t) * x[0] ** 3
    return dxdt

diff(F(x, t, u), x)










# def f1(T2, T4, A):
#
#     dimension, _ = T2.shape
#
#     phi = np.zeros([dimension, dimension, dimension, dimension, dimension])
#
#     for i in range(dimension):
#         for j1 in range(dimension):
#             for j2 in range(dimension):
#                 for j3 in range(dimension):
#                     for j4 in range(dimension):
#                         for r1 in range(dimension):
#                             for r2 in range(dimension):
#                                 phi[i, j1, j2, j3, j4] += A[i, r1, r2] * T4[r1, j1, j3, j4] * T2[r2, j2]
#
#     return phi
#
#
# def f2(T2, T4, A):
#
#     phi = np.transpose(np.tensordot(A, np.tensordot(T4, T2, axes=0), axes=([1, 2], [0, 4])), axes=[0, 1, 4, 2, 3])
#     # phii = np.tensordot(A, np.transpose(np.tensordot(T4, T2, axes=0), axes=[0, 1, 3, 5, 4, 2]), axes=([1, 2], [0, 4]))
#     # phi = np.tensordot(A, np.tensordot(T4, T2, axes=0), axes=([1, 2], [0, 4]))
#
#     return phi
#
#
# A = np.random.randn(8).reshape(2, 2, 2)
# T2 = np.random.randn(4).reshape(2, 2)
# T4 = np.random.randn(16).reshape(2, 2, 2, 2)
#
# phi1 = f1(T2, T4, A)
# phi2 = f2(T2, T4, A)
#
# print(phi1 - phi2)
