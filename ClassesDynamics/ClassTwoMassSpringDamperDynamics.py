"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm


class TwoMassSpringDamperDynamics:
    def __init__(self, dt, mass1, mass2, spring_constant1, spring_constant2, damping_coefficient1, damping_coefficient2, inputs, measurements1, measurements2):
        self.state_dimension = 4
        self.input_dimension = max(1, len(inputs))
        self.output_dimension = len(measurements1) + len(measurements2)
        self.dt = dt
        self.frequency = 1 / dt
        self.mass1 = mass1
        self.mass2 = mass2
        self.spring_constant1 = spring_constant1
        self.spring_constant2 = spring_constant2
        self.damping_coefficient1 = damping_coefficient1
        self.damping_coefficient2 = damping_coefficient2
        self.inputs = inputs
        self.measurements1 = measurements1
        self.measurements2 = measurements2
        self.total_measurements = []
        self.M = np.zeros([2, 2])
        self.K = np.zeros([2, 2])
        self.Z = np.zeros([2, 2])
        self.M[0, 0] = self.mass1
        self.M[1, 1] = self.mass2
        self.K[0, 0] = self.spring_constant1 + self.spring_constant2
        self.K[0, 1] = -self.spring_constant2
        self.K[1, 0] = -self.spring_constant2
        self.K[1, 1] = self.spring_constant2
        self.Z[0, 0] = self.damping_coefficient1 + self.damping_coefficient2
        self.Z[0, 1] = -self.damping_coefficient2
        self.Z[1, 0] = -self.damping_coefficient2
        self.Z[1, 1] = self.damping_coefficient2

        self.Ac = np.zeros([self.state_dimension, self.state_dimension])
        self.Ac[0:2, 2:4] = np.eye(2)
        self.Ac[2:4, 0:2] = np.matmul(-inv(self.M), self.K)
        self.Ac[2:4, 2:4] = np.matmul(-inv(self.M), self.Z)
        self.Ad = expm(self.Ac * self.dt)

        n2 = int(self.state_dimension / 2)
        self.B2 = np.zeros([n2, self.input_dimension])
        self.initial_condition_response = True
        i = 0
        if 'mass1' in self.inputs:
            self.B2[0, i] = 1
            self.initial_condition_response = False
            i += 1
        if 'mass2' in self.inputs:
            self.B2[1, i] = 1
            self.initial_condition_response = False
            i += 1
        self.Bc = np.zeros([self.state_dimension, self.input_dimension])
        self.Bc[2:4, 0:2] = np.matmul(inv(self.M), self.B2)
        self.Bd = np.matmul(np.matmul((self.Ad - np.eye(self.state_dimension)), inv(self.Ac)), self.Bc)

        self.Cd = np.zeros([self.output_dimension, self.state_dimension])
        self.Cp = np.zeros([self.output_dimension, int(self.state_dimension / 2)])
        self.Cv = np.zeros([self.output_dimension, int(self.state_dimension / 2)])
        self.Ca = np.zeros([self.output_dimension, int(self.state_dimension / 2)])
        i = 0
        if 'position' in self.measurements1:
            self.Cp[i, 0] = 1
            i += 1
            self.total_measurements.append('Position 1')
        if 'position' in self.measurements2:
            self.Cp[i, 1] = 1
            i += 1
            self.total_measurements.append('Position 2')
        if 'velocity' in self.measurements1:
            self.Cv[i, 0] = 1
            i += 1
            self.total_measurements.append('Velocity 1')
        if 'velocity' in self.measurements2:
            self.Cv[i, 1] = 1
            i += 1
            self.total_measurements.append('Velocity 2')
        if 'acceleration' in self.measurements1:
            self.Ca[i, 0] = 1
            i += 1
            self.total_measurements.append('Acceleration 1')
        if 'acceleration' in self.measurements2:
            self.Ca[i, 1] = 1
            i += 1
            self.total_measurements.append('Acceleration 2')
        self.Cd[:, 0:int(self.state_dimension / 2)] = self.Cp - np.matmul(self.Ca, np.matmul(inv(self.M), self.K))
        self.Cd[:, int(self.state_dimension / 2): self.state_dimension] = self.Cv - np.matmul(self.Ca, np.matmul(inv(self.M), self.Z))

        self.Dd = np.matmul(self.Ca, np.matmul(inv(self.M), self.B2))


    def A(self, tk):
        return self.Ad

    def B(self, tk):
        return self.Bd

    def C(self, tk):
        return self.Cd

    def D(self, tk):
        return self.Dd
