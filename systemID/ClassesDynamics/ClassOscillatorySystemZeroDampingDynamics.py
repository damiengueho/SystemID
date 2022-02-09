"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""



import numpy as np
from scipy.linalg import expm


class OscillatorySystemZeroDampingDynamics:
    def __init__(self, dt):
        self.state_dimension = 4
        self.input_dimension = 2
        self.output_dimension = 2
        self.dt = dt

    def Ac(self, tk):
        Ac = np.zeros([4, 4])
        c = np.cos(10 * tk)
        s = np.sin(10 * tk)
        Ac[0, 2] = 1
        Ac[1, 3] = 1
        Ac[2, 0] = -4 - 3 * s
        Ac[2, 1] = -1
        Ac[3, 0] = -1
        Ac[3, 1] = -7 - 3 * c
        return Ac

    def A(self, tk):
        return expm(self.Ac(tk) * self.dt)

    def B(self, tk):
        B = np.zeros([4, 2])
        B[0, 0] = 1
        B[1, 0] = 1
        B[1, 1] = -1
        B[2, 1] = 1
        B[3, 0] = -1
        return B

    def C(self, tk):
        C = np.zeros([2, 4])
        C[0, 0] = 1
        C[0, 2] = 1
        C[0, 3] = 0.2
        C[1, 0] = 1
        C[1, 1] = -1
        C[1, 3] = -0.5
        return C

    def D(self, tk):
        return 0.1 * np.eye(2)
