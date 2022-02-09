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


class BallAndBeamDynamics:
    def __init__(self, dt, ball_mass, ball_radius, lever_arm_offset, beam_length, ball_moment_of_inertia, inputs, measurements):
        self.state_dimension = 4
        self.input_dimension = 1
        self.output_dimension = min(1, len(measurements))
        self.dt = dt
        self.frequency = 1 / dt
        self.ball_mass = ball_mass
        self.ball_radius = ball_radius
        self.lever_arm_offset = lever_arm_offset
        self.beam_length = beam_length
        self.ball_moment_of_inertia = ball_moment_of_inertia
        self.inputs = inputs
        self.measurements = measurements
        self.total_measurements = []

        self.Ac = np.zeros([self.state_dimension, self.state_dimension])
        self.Ac[0, 1] = 1
        self.Ac[1, 2] = -self.ball_mass * 9.81 / (self.ball_moment_of_inertia / (self.ball_radius ** 2) + self.ball_mass)
        self.Ac[2, 3] = 1
        self.Ad = expm(self.Ac * self.dt)

        self.Bc = np.zeros([self.state_dimension, self.input_dimension])
        self.initial_condition_response = True
        if 'gear angle' in self.inputs:
            self.Bc[3, 0] = 1
            self.initial_condition_response = False
        self.Bd = np.matmul(np.matmul((self.Ad - np.eye(self.state_dimension)), inv(self.Ac)), self.Bc)

        self.Cd = np.zeros([self.output_dimension, self.state_dimension])
        i = 0
        j = 0
        if 'position' in self.measurements:
            self.Cd[i, j] = 1
            i += 1
            self.total_measurements.append('Ball position')
        j += 1
        if 'velocity' in self.measurements:
            self.Cd[i, j] = 1
            i += 1
            self.total_measurements.append('Ball velocity')
        j += 1
        if 'angular position' in self.measurements:
            self.Cd[i, j] = 1
            i += 1
            self.total_measurements.append('Angular position')
        j += 1
        if 'angular velocity' in self.measurements:
            self.Cd[i, j] = 1
            i += 1
            self.total_measurements.append('Angular velocity')

        self.Dd = np.zeros([self.output_dimension, self.input_dimension])


    def A(self, tk):
        return self.Ad

    def B(self, tk):
        return self.Bd

    def C(self, tk):
        return self.Cd

    def D(self, tk):
        return self.Dd
