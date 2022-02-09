"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""


import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm


class MassSpringDamperDynamics:
    def __init__(self, dt, mass, spring_constant, damping_coefficient, inputs, measurements):
        self.state_dimension = 2
        self.input_dimension = 1
        self.output_dimension = len(measurements)
        self.dt = dt
        self.frequency = 1 / dt
        self.mass = mass
        self.spring_constant = spring_constant
        self.damping_coefficient = damping_coefficient
        self.inputs = inputs
        self.measurements = measurements
        self.total_measurements = []
        self.M = np.zeros([1, 1])
        self.K = np.zeros([1, 1])
        self.Z = np.zeros([1, 1])
        self.M[0, 0] = self.mass
        self.K[0, 0] = self.spring_constant
        self.Z[0, 0] = self.damping_coefficient

        self.Ac = np.zeros([self.state_dimension, self.state_dimension])
        self.Ac[0:1, 1:2] = 1
        self.Ac[1:2, 0:1] = np.matmul(-inv(self.M), self.K)
        self.Ac[1:2, 1:2] = np.matmul(-inv(self.M), self.Z)
        self.Ad = expm(self.Ac * self.dt)

        self.B2 = np.zeros([int(self.state_dimension / 2), self.input_dimension])
        self.initial_condition_response = True
        if 'mass' in self.inputs:
            self.B2[0, 0] = 1
            self.initial_condition_response = False
        self.Bc = np.zeros([self.state_dimension, self.input_dimension])
        self.Bc[1:2, 0:1] = np.matmul(inv(self.M), self.B2)
        self.Bd = np.matmul(np.matmul((self.Ad - np.eye(self.state_dimension)), inv(self.Ac)), self.Bc)

        self.Cd = np.zeros([self.output_dimension, self.state_dimension])
        self.Cp = np.zeros([self.output_dimension, int(self.state_dimension / 2)])
        self.Cv = np.zeros([self.output_dimension, int(self.state_dimension / 2)])
        self.Ca = np.zeros([self.output_dimension, int(self.state_dimension / 2)])
        i = 0
        if 'position' in self.measurements:
            self.Cp[i, 0] = 1
            i += 1
            self.total_measurements.append('Position')
        if 'velocity' in self.measurements:
            self.Cv[i, 0] = 1
            i += 1
            self.total_measurements.append('Velocity')
        if 'acceleration' in self.measurements:
            self.Ca[i, 0] = 1
            i += 1
            self.total_measurements.append('Acceleration')
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



# spring constant is seen as an input
class MassSpringDamperDynamicsBilinear:
    def __init__(self, mass, damping_coefficient, measurements):
        self.state_dimension = 2
        self.input_dimension = 1
        self.output_dimension = len(measurements)
        self.mass = mass
        self.damping_coefficient = damping_coefficient
        self.measurements = measurements
        self.total_measurements = []
        self.units = []
        self.M = np.zeros([1, 1])
        self.K = np.zeros([1, 1])
        self.Z = np.zeros([1, 1])
        self.M[0, 0] = self.mass
        self.Z[0, 0] = self.damping_coefficient

        self.Ac = np.zeros([self.state_dimension, self.state_dimension])
        self.Ac[0:1, 1:2] = 1
        self.Ac[1:2, 1:2] = np.matmul(-inv(self.M), self.Z)

        self.Nc = np.zeros([self.state_dimension, self.state_dimension * self.input_dimension])
        self.Nc[1, 0] = -1 / self.mass

        self.Bc = np.zeros([self.state_dimension, self.input_dimension])

        self.Cd = np.zeros([self.output_dimension, self.state_dimension])
        self.Cp = np.zeros([self.output_dimension, int(self.state_dimension / 2)])
        self.Cv = np.zeros([self.output_dimension, int(self.state_dimension / 2)])
        self.Ca = np.zeros([self.output_dimension, int(self.state_dimension / 2)])
        i = 0
        if 'position' in self.measurements:
            self.Cp[i, 0] = 1
            i += 1
            self.total_measurements.append('Position')
            self.units.append('m')
        if 'velocity' in self.measurements:
            self.Cv[i, 0] = 1
            i += 1
            self.total_measurements.append('Velocity')
            self.units.append('m/s')
        if 'acceleration' in self.measurements:
            self.Ca[i, 0] = 1
            i += 1
            self.total_measurements.append('Acceleration')
            self.units.append('m/s^2')
        self.Cd[:, 0:int(self.state_dimension / 2)] = self.Cp - np.matmul(self.Ca, np.matmul(inv(self.M), self.K))
        self.Cd[:, int(self.state_dimension / 2): self.state_dimension] = self.Cv - np.matmul(self.Ca, np.matmul(inv(self.M), self.Z))

        self.Dd = np.zeros([self.output_dimension, self.input_dimension])


    def A(self, tk):
        return self.Ac

    def N(self, tk):
        return self.Nc

    def B(self, tk):
        return self.Bc

    def C(self, tk):
        return self.Cd

    def D(self, tk):
        return self.Dd
