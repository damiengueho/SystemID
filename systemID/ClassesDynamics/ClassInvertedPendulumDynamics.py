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


class InvertedPendulumDynamics:
    def __init__(self, dt, cart_mass, pendulum_mass, cart_friction_coefficient, length_to_pendulum_center_of_mass, pendulum_mass_moment_of_inertia, inputs, measurements_cart, measurements_beam):
        self.state_dimension = 4
        self.input_dimension = 1
        self.output_dimension = min(1, len(measurements_cart) + len(measurements_beam))
        self.dt = dt
        self.frequency = 1 / dt
        self.cart_mass = cart_mass
        self.pendulum_mass = pendulum_mass
        self.cart_friction_coefficient = cart_friction_coefficient
        self.length_to_pendulum_center_of_mass = length_to_pendulum_center_of_mass
        self.pendulum_mass_moment_of_inertia = pendulum_mass_moment_of_inertia
        self.inputs = inputs
        self.measurements_cart = measurements_cart
        self.measurements_beam = measurements_beam
        self.total_measurements = []

        self.a = self.pendulum_mass_moment_of_inertia + self.pendulum_mass * self.length_to_pendulum_center_of_mass ** 2
        self.b = self.pendulum_mass_moment_of_inertia * (self.cart_mass + self.pendulum_mass)
        self.c = self.cart_mass * self.pendulum_mass * self.length_to_pendulum_center_of_mass ** 2
        self.d = self.pendulum_mass * 9.81 * self.length_to_pendulum_center_of_mass
        self.e = self.pendulum_mass * self.length_to_pendulum_center_of_mass

        self.Ac = np.zeros([self.state_dimension, self.state_dimension])
        self.Ac[0, 1] = 1
        self.Ac[1, 1] = -self.a * self.cart_friction_coefficient / (self.b + self.c)
        self.Ac[1, 2] = self.d ** 2 / (self.b + self.c)
        self.Ac[2, 3] = 1
        self.Ac[3, 1] = -self.e * self.cart_friction_coefficient / (self.b + self.c)
        self.Ac[3, 2] = self.d * (self.cart_mass + self.pendulum_mass) / (self.b + self.c)
        self.Ad = expm(self.Ac * self.dt)

        self.Bc = np.zeros([self.state_dimension, self.input_dimension])
        self.initial_condition_response = True
        if 'cart' in self.inputs:
            self.Bc[1, 0] = self.a / (self.b + self.c)
            self.Bc[3, 0] = self.e / (self.b + self.c)
            self.initial_condition_response = False
        self.Bd = np.matmul(np.matmul((self.Ad - np.eye(self.state_dimension)), inv(self.Ac)), self.Bc)

        self.Cd = np.zeros([self.output_dimension, self.state_dimension])
        i = 0
        j = 0
        if 'position' in self.measurements_cart:
            self.Cd[i, j] = 1
            i += 1
            self.total_measurements.append('Cart position')
        j += 1
        if 'velocity' in self.measurements_cart:
            self.Cd[i, j] = 1
            i += 1
            self.total_measurements.append('Cart velocity')
        j += 1
        if 'angular position' in self.measurements_beam:
            self.Cd[i, j] = 1
            i += 1
            self.total_measurements.append('Angular position')
        j += 1
        if 'angular velocity' in self.measurements_beam:
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
