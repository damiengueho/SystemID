"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""



import numpy as np


class SystemWithAStableOriginDynamics:
    def __init__(self):
        self.state_dimension = 3
        self.input_dimension = 2
        self.output_dimension = 2

    def A(self, tk):
        A = np.zeros([3, 3])
        c = np.cos(10 * tk)
        s = np.sin(10 * tk)
        A[0, 0] = 0.3 - 0.9 * s
        A[0, 1] = 0.1
        A[0, 2] = 0.7 * c
        A[1, 0] = 0.6 * s
        A[1, 1] = 0.3 - 0.8 * c
        A[1, 2] = 0.01
        A[2, 0] = 0.5
        A[2, 1] = 0.15
        A[2, 2] = 0.6 - 0.9 * s
        return A

    def B(self, tk):
        B = np.zeros([3, 2])
        B[0, 0] = 1
        B[1, 0] = 1
        B[1, 1] = -1
        B[2, 1] = 1
        return B

    def C(self, tk):
        C = np.zeros([2, 3])
        C[0, 0] = 1
        C[0, 2] = 1
        C[1, 0] = 1
        C[1, 1] = -1
        return C

    def D(self, tk):
        return 0.1 * np.eye(2)
        # return np.zeros([2, 2])
