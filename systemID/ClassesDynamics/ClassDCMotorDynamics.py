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


class DCMotorDynamics:
    def __init__(self, dt, electric_resistance, electric_inductance, rotor_moment_inertia, rotor_viscous_friction_constant, rotor_torque_constant, inputs, measurements_rotor, measurements_circuit):
        self.state_dimension = 3
        self.input_dimension = 1
        self.output_dimension = min(1, len(measurements_rotor) + len(measurements_circuit))
        self.dt = dt
        self.frequency = 1 / dt
        self.electric_resistance = electric_resistance
        self.electric_inductance = electric_inductance
        self.rotor_moment_inertia = rotor_moment_inertia
        self.rotor_viscous_friction_constant = rotor_viscous_friction_constant
        self.rotor_torque_constant = rotor_torque_constant
        self.inputs = inputs
        self.measurements_rotor = measurements_rotor
        self.measurements_circuit = measurements_circuit
        self.total_measurements = []

        self.Ac = np.zeros([self.state_dimension, self.state_dimension])
        self.Ac[0, 1] = 1
        self.Ac[1, 1] = -self.rotor_viscous_friction_constant / self.rotor_moment_inertia
        self.Ac[1, 2] = self.rotor_torque_constant / self.rotor_moment_inertia
        self.Ac[2, 1] = -self.rotor_torque_constant / self.electric_inductance
        self.Ac[2, 2] = -self.electric_resistance / self.electric_inductance
        self.Ad = expm(self.Ac * self.dt)

        self.Bc = np.zeros([self.state_dimension, self.input_dimension])
        self.initial_condition_response = True
        if 'voltage source' in self.inputs:
            self.Bc[2, 0] = 1 / self.electric_inductance
            self.initial_condition_response = False
        self.Bd = np.matmul(np.matmul((self.Ad - np.eye(self.state_dimension)), inv(self.Ac)), self.Bc)

        self.Cd = np.zeros([self.output_dimension, self.state_dimension])
        self.Dd = np.zeros([self.output_dimension, self.input_dimension])
        i = 0
        if 'angular position' in self.measurements_rotor:
            self.Cd[i, 0] = 1
            i += 1
            self.total_measurements.append('Angular position')
        if 'angular velocity' in self.measurements_rotor:
            self.Cd[i, 1] = 1
            i += 1
            self.total_measurements.append('Angular velocity')
        if 'angular acceleration' in self.measurements_rotor:
            self.Cd[i, 1] = -self.rotor_viscous_friction_constant / self.rotor_moment_inertia
            self.Cd[i, 2] = self.rotor_torque_constant / self.rotor_moment_inertia
            i += 1
            self.total_measurements.append('Angular acceleration')
        if 'current' in self.measurements_circuit:
            self.Cd[i, 2] = 1
            i += 1
            self.total_measurements.append('Current')
        if 'current speed' in self.measurements_circuit:
            self.Cd[i, 1] = -self.rotor_torque_constant / self.electric_inductance
            self.Cd[i, 2] = -self.electric_resistance / self.electric_inductance
            self.Dd[i, 0] = 1 / self.electric_inductance
            i += 1
            self.total_measurements.append('Current speed')




    def A(self, tk):
        return self.Ad

    def B(self, tk):
        return self.Bd

    def C(self, tk):
        return self.Cd

    def D(self, tk):
        return self.Dd
