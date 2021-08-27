"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 15
Date: August 2021
Python: 3.7.7
"""



import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from ClassesGeneral.ClassSignal import DiscreteSignal


class DuffingOscillatorDynamics:
    """
    Dynamics
    x1' = x2
    x2' = -delta*x2 - alpha*x1 - beta*x1^3 + u
    """

    def __init__(self, delta, alpha, beta, **kwargs):
        self.state_dimension = 2
        self.input_dimension = 1
        self.output_dimension = 2
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.tspan = kwargs.get('tspan', np.array([0, 1, 2, 3]))
        self.nominal_x = kwargs.get('nominal_x', DiscreteSignal(self.state_dimension, 3, 1))
        self.nominal_x_interpolated = interp1d(self.tspan, self.nominal_x.data, 'cubic')
        self.nominal_u = kwargs.get('nominal_u', DiscreteSignal(self.input_dimension, 3, 1))
        self.nominal_u_interpolated = interp1d(self.tspan, self.nominal_u.data, 'cubic')
        self.dt = kwargs.get('dt', 0)

    def F(self, x, t, u):
        dxdt = np.zeros(self.state_dimension)
        dxdt[0] = x[1]
        dxdt[1] = -self.delta(t) * x[1] - self.alpha(t) * x[0] - self.beta(t) * x[0] ** 3 + u(t)
        return dxdt

    def G(self, x, t, u):
        return x

    def Ac(self, t):
        Ac = np.zeros([self.state_dimension, self.state_dimension])
        Ac[0, 1] = 1
        Ac[1, 0] = -self.alpha(t) - 3 * self.beta(t) * self.nominal_x_interpolated(t)[0] ** 2
        Ac[1, 1] = -self.delta(t)
        return Ac

    def dPhi(self, Phi, t):
        return np.matmul(self.Ac(t), Phi.reshape(self.state_dimension, self.state_dimension)).reshape(self.state_dimension**2)

    def A(self, tk):
        A = odeint(self.dPhi, np.eye(self.state_dimension).reshape(self.state_dimension**2), np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return A[-1, :].reshape(self.state_dimension, self.state_dimension)

    def Bc(self, t):
        Bc = np.zeros([self.state_dimension, self.input_dimension])
        Bc[1, 0] = 1
        return Bc

    def dPsi(self, Psi, t):
        return np.matmul(self.Ac(t), Psi.reshape(self.state_dimension, self.state_dimension)).reshape(self.state_dimension**2) + np.eye(self.state_dimension).reshape(self.state_dimension**2)

    def B(self, tk):
        B = odeint(self.dPsi, np.zeros([self.state_dimension, self.state_dimension]).reshape(self.state_dimension**2), np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return np.matmul(B[-1, :].reshape(self.state_dimension, self.state_dimension), self.Bc(tk))

    def C(self, tk):
        C = np.eye(self.state_dimension)
        return C

    def D(self, tk):
        D = np.zeros([self.output_dimension, self.input_dimension])
        return D










class DuffingOscillatorDynamics2:
    """
    Parameters are treated as input
    """

    def __init__(self, **kwargs):
        self.state_dimension = 2
        self.input_dimension = 3
        self.output_dimension = 2
        self.tspan = kwargs.get('tspan', np.array([0, 1, 2, 3]))
        self.nominal_x = kwargs.get('nominal_x', DiscreteSignal(self.state_dimension, 'No nominal trajectory', 3, 1))
        self.nominal_x_interpolated = interp1d(self.tspan, self.nominal_x.data, 'cubic')
        self.nominal_u = kwargs.get('nominal_u', DiscreteSignal(self.input_dimension, 'No nominal input', 3, 1))
        self.nominal_u_interpolated = interp1d(self.tspan, self.nominal_u.data, 'cubic')
        self.dt = kwargs.get('dt', 0)

    def F(self, x, t, u):
        dxdt = np.zeros(self.state_dimension)
        dxdt[0] = x[1]
        dxdt[1] = -u(t)[0] * x[1] - u(t)[1] * x[0] - u(t)[2] * x[0] ** 3
        return dxdt

    def G(self, x, t, u):
        return x

    def Ac(self, t):
        Ac = np.zeros([self.state_dimension, self.state_dimension])
        Ac[0, 1] = 1
        Ac[1, 0] = -self.nominal_u_interpolated(t)[1] - 3 * self.nominal_u_interpolated(t)[2] * self.nominal_x_interpolated(t)[0] ** 2
        Ac[1, 1] = -self.nominal_u_interpolated(t)[0]
        return Ac

    def dPhi(self, Phi, t):
        return np.matmul(self.Ac(t), Phi.reshape(self.state_dimension, self.state_dimension)).reshape(self.state_dimension**2)

    def A(self, tk):
        A = odeint(self.dPhi, np.eye(self.state_dimension).reshape(self.state_dimension**2), np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return A[-1, :].reshape(self.state_dimension, self.state_dimension)

    def Bc(self, t):
        Bc = np.zeros([self.state_dimension, self.input_dimension])
        Bc[1, 0] = -self.nominal_x_interpolated(t)[1]
        Bc[1, 1] = -self.nominal_x_interpolated(t)[0]
        Bc[1, 2] = -self.nominal_x_interpolated(t)[0] ** 3
        return Bc

    def dPsi(self, Psi, t):
        return np.matmul(self.Ac(t), Psi.reshape(self.state_dimension, self.state_dimension)).reshape(self.state_dimension**2) + np.eye(self.state_dimension).reshape(self.state_dimension**2)

    def B(self, tk):
        B = odeint(self.dPsi, np.zeros([self.state_dimension, self.state_dimension]).reshape(self.state_dimension**2), np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return np.matmul(B[-1, :].reshape(self.state_dimension, self.state_dimension), self.Bc(tk))

    def C(self, tk):
        C = np.eye(self.state_dimension)
        return C

    def D(self, tk):
        D = np.zeros([self.output_dimension, self.input_dimension])
        return D









class DuffingOscillatorDynamics3:
    """
    Only delta is considered as an input
    """

    def __init__(self, alpha, beta, **kwargs):
        self.state_dimension = 2
        self.input_dimension = 1
        self.output_dimension = 2
        self.alpha = alpha
        self.beta = beta
        self.tspan = kwargs.get('tspan', np.array([0, 1, 2, 3]))
        self.nominal_x = kwargs.get('nominal_x', DiscreteSignal(self.state_dimension, 'No nominal trajectory', 3, 1))
        self.nominal_x_interpolated = interp1d(self.tspan, self.nominal_x.data, 'cubic')
        self.nominal_u = kwargs.get('nominal_u', DiscreteSignal(self.input_dimension, 'No nominal input', 3, 1))
        self.nominal_u_interpolated = interp1d(self.tspan, self.nominal_u.data, 'cubic')
        self.dt = kwargs.get('dt', 0)

    def F(self, x, t, u):
        dxdt = np.zeros(self.state_dimension)
        dxdt[0] = x[1]
        dxdt[1] = -u(t) * x[1] - self.alpha(t) * x[0] - self.beta(t) * x[0] ** 3
        return dxdt

    def G(self, x, t, u):
        return x

    def Ac(self, t):
        Ac = np.zeros([self.state_dimension, self.state_dimension])
        Ac[0, 1] = 1
        Ac[1, 0] = -self.alpha(t) - 3 * self.beta(t) * self.nominal_x_interpolated(t)[0] ** 2
        Ac[1, 1] = -self.nominal_u_interpolated(t)
        return Ac

    def dPhi(self, Phi, t):
        return np.matmul(self.Ac(t), Phi.reshape(self.state_dimension, self.state_dimension)).reshape(
            self.state_dimension ** 2)

    def A(self, tk):
        A = odeint(self.dPhi, np.eye(self.state_dimension).reshape(self.state_dimension ** 2),
                   np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return A[-1, :].reshape(self.state_dimension, self.state_dimension)

    def Bc(self, t):
        Bc = np.zeros([self.state_dimension, self.input_dimension])
        Bc[1, 0] = -self.nominal_x_interpolated(t)[1]
        return Bc

    def dPsi(self, Psi, t):
        return np.matmul(self.Ac(t), Psi.reshape(self.state_dimension, self.state_dimension)).reshape(
            self.state_dimension ** 2) + np.eye(self.state_dimension).reshape(self.state_dimension ** 2)

    def B(self, tk):
        B = odeint(self.dPsi, np.zeros([self.state_dimension, self.state_dimension]).reshape(self.state_dimension ** 2),
                   np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return np.matmul(B[-1, :].reshape(self.state_dimension, self.state_dimension), self.Bc(tk))

    def C(self, tk):
        C = np.eye(self.state_dimension)
        return C

    def D(self, tk):
        D = np.zeros([self.output_dimension, self.input_dimension])
        return D









class DuffingOscillatorDynamics4:
    """
    Only alpha is considered as an input
    """

    def __init__(self, delta, beta, **kwargs):
        self.state_dimension = 2
        self.input_dimension = 1
        self.output_dimension = 2
        self.delta = delta
        self.beta = beta
        self.tspan = kwargs.get('tspan', np.array([0, 1, 2, 3]))
        self.nominal_x = kwargs.get('nominal_x', DiscreteSignal(self.state_dimension, 'No nominal trajectory', 3, 1))
        self.nominal_x_interpolated = interp1d(self.tspan, self.nominal_x.data, 'cubic')
        self.nominal_u = kwargs.get('nominal_u', DiscreteSignal(self.input_dimension, 'No nominal input', 3, 1))
        self.nominal_u_interpolated = interp1d(self.tspan, self.nominal_u.data, 'cubic')
        self.dt = kwargs.get('dt', 0)

    def F(self, x, t, u):
        dxdt = np.zeros(self.state_dimension)
        dxdt[0] = x[1]
        dxdt[1] = -self.delta(t) * x[1] - u(t) * x[0] - self.beta(t) * x[0] ** 3
        return dxdt

    def G(self, x, t, u):
        return x

    def Ac(self, t):
        Ac = np.zeros([self.state_dimension, self.state_dimension])
        Ac[0, 1] = 1
        Ac[1, 0] = -self.nominal_u_interpolated(t) - 3 * self.beta(t) * self.nominal_x_interpolated(t)[0] ** 2
        Ac[1, 1] = -self.delta(t)
        return Ac

    def dPhi(self, Phi, t):
        return np.matmul(self.Ac(t), Phi.reshape(self.state_dimension, self.state_dimension)).reshape(
            self.state_dimension ** 2)

    def A(self, tk):
        A = odeint(self.dPhi, np.eye(self.state_dimension).reshape(self.state_dimension ** 2),
                   np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return A[-1, :].reshape(self.state_dimension, self.state_dimension)

    def Bc(self, t):
        Bc = np.zeros([self.state_dimension, self.input_dimension])
        Bc[1, 0] = -self.nominal_x_interpolated(t)[0]
        return Bc

    def dPsi(self, Psi, t):
        return np.matmul(self.Ac(t), Psi.reshape(self.state_dimension, self.state_dimension)).reshape(
            self.state_dimension ** 2) + np.eye(self.state_dimension).reshape(self.state_dimension ** 2)

    def B(self, tk):
        B = odeint(self.dPsi, np.zeros([self.state_dimension, self.state_dimension]).reshape(self.state_dimension ** 2),
                   np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return np.matmul(B[-1, :].reshape(self.state_dimension, self.state_dimension), self.Bc(tk))

    def C(self, tk):
        C = np.eye(self.state_dimension)
        return C

    def D(self, tk):
        D = np.zeros([self.output_dimension, self.input_dimension])
        return D










class DuffingOscillatorDynamics5:
    """
    Only beta is considered as an input
    """

    def __init__(self, delta, alpha, **kwargs):
        self.state_dimension = 2
        self.input_dimension = 1
        self.output_dimension = 2
        self.delta = delta
        self.alpha = alpha
        self.tspan = kwargs.get('tspan', np.array([0, 1, 2, 3]))
        self.nominal_x = kwargs.get('nominal_x', DiscreteSignal(self.state_dimension, 'No nominal trajectory', 3, 1))
        self.nominal_x_interpolated = interp1d(self.tspan, self.nominal_x.data, 'cubic')
        self.nominal_u = kwargs.get('nominal_u', DiscreteSignal(self.input_dimension, 'No nominal input', 3, 1))
        self.nominal_u_interpolated = interp1d(self.tspan, self.nominal_u.data, 'cubic')
        self.dt = kwargs.get('dt', 0)

    def F(self, x, t, u):
        dxdt = np.zeros(self.state_dimension)
        dxdt[0] = x[1]
        dxdt[1] = -self.delta(t) * x[1] - self.alpha(t) * x[0] - u(t) * x[0] ** 3
        return dxdt

    def G(self, x, t, u):
        return x

    def Ac(self, t):
        Ac = np.zeros([self.state_dimension, self.state_dimension])
        Ac[0, 1] = 1
        Ac[1, 0] = -self.alpha(t) - 3 * self.nominal_u_interpolated(t) * self.nominal_x_interpolated(t)[0] ** 2
        Ac[1, 1] = -self.delta(t)
        return Ac

    def dPhi(self, Phi, t):
        return np.matmul(self.Ac(t), Phi.reshape(self.state_dimension, self.state_dimension)).reshape(
            self.state_dimension ** 2)

    def A(self, tk):
        A = odeint(self.dPhi, np.eye(self.state_dimension).reshape(self.state_dimension ** 2),
                   np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return A[-1, :].reshape(self.state_dimension, self.state_dimension)

    def Bc(self, t):
        Bc = np.zeros([self.state_dimension, self.input_dimension])
        Bc[1, 0] = -self.nominal_x_interpolated(t)[0] ** 3
        return Bc

    def dPsi(self, Psi, t):
        return np.matmul(self.Ac(t), Psi.reshape(self.state_dimension, self.state_dimension)).reshape(
            self.state_dimension ** 2) + np.eye(self.state_dimension).reshape(self.state_dimension ** 2)

    def B(self, tk):
        B = odeint(self.dPsi, np.zeros([self.state_dimension, self.state_dimension]).reshape(self.state_dimension ** 2),
                   np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return np.matmul(B[-1, :].reshape(self.state_dimension, self.state_dimension), self.Bc(tk))

    def C(self, tk):
        C = np.eye(self.state_dimension)
        return C

    def D(self, tk):
        D = np.zeros([self.output_dimension, self.input_dimension])
        return D










class DuffingOscillatorDynamics6:
    """
    delta and alpha are considered as inputs
    """

    def __init__(self, beta, **kwargs):
        self.state_dimension = 2
        self.input_dimension = 2
        self.output_dimension = 2
        self.beta = beta
        self.tspan = kwargs.get('tspan', np.array([0, 1, 2, 3]))
        self.nominal_x = kwargs.get('nominal_x', DiscreteSignal(self.state_dimension, 'No nominal trajectory', 3, 1))
        self.nominal_x_interpolated = interp1d(self.tspan, self.nominal_x.data, 'cubic')
        self.nominal_u = kwargs.get('nominal_u', DiscreteSignal(self.input_dimension, 'No nominal input', 3, 1))
        self.nominal_u_interpolated = interp1d(self.tspan, self.nominal_u.data, 'cubic')
        self.dt = kwargs.get('dt', 0)

    def F(self, x, t, u):
        dxdt = np.zeros(self.state_dimension)
        dxdt[0] = x[1]
        dxdt[1] = -u(t)[0] * x[1] - u(t)[1] * x[0] - self.beta(t) * x[0] ** 3
        return dxdt

    def G(self, x, t, u):
        return x

    def Ac(self, t):
        Ac = np.zeros([self.state_dimension, self.state_dimension])
        Ac[0, 1] = 1
        Ac[1, 0] = -self.nominal_u_interpolated(t)[1] - 3 * self.beta(t) * self.nominal_x_interpolated(t)[0] ** 2
        Ac[1, 1] = -self.nominal_u_interpolated(t)[0]
        return Ac

    def dPhi(self, Phi, t):
        return np.matmul(self.Ac(t), Phi.reshape(self.state_dimension, self.state_dimension)).reshape(
            self.state_dimension ** 2)

    def A(self, tk):
        A = odeint(self.dPhi, np.eye(self.state_dimension).reshape(self.state_dimension ** 2),
                   np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return A[-1, :].reshape(self.state_dimension, self.state_dimension)

    def Bc(self, t):
        Bc = np.zeros([self.state_dimension, self.input_dimension])
        Bc[1, 0] = -self.nominal_x_interpolated(t)[1]
        Bc[1, 1] = -self.nominal_x_interpolated(t)[0]
        return Bc

    def dPsi(self, Psi, t):
        return np.matmul(self.Ac(t), Psi.reshape(self.state_dimension, self.state_dimension)).reshape(
            self.state_dimension ** 2) + np.eye(self.state_dimension).reshape(self.state_dimension ** 2)

    def B(self, tk):
        B = odeint(self.dPsi, np.zeros([self.state_dimension, self.state_dimension]).reshape(self.state_dimension ** 2),
                   np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return np.matmul(B[-1, :].reshape(self.state_dimension, self.state_dimension), self.Bc(tk))

    def C(self, tk):
        C = np.eye(self.state_dimension)
        return C

    def D(self, tk):
        D = np.zeros([self.output_dimension, self.input_dimension])
        return D










class DuffingOscillatorDynamics7:
    """
    True system is discrete-time, parameters are inputs
    """

    def __init__(self, dt, **kwargs):
        self.state_dimension = 2
        self.input_dimension = 3
        self.output_dimension = 2
        self.dt = dt
        self.tspan = kwargs.get('tspan', np.array([0, 1, 2, 3]))
        self.nominal_x = kwargs.get('nominal_x', DiscreteSignal(self.state_dimension, 'No nominal trajectory', 3, 1))
        self.nominal_x_interpolated = interp1d(self.tspan, self.nominal_x.data, 'cubic')
        self.nominal_u = kwargs.get('nominal_u', DiscreteSignal(self.input_dimension, 'No nominal input', 3, 1))
        self.nominal_u_interpolated = interp1d(self.tspan, self.nominal_u.data, 'cubic')

    def F(self, x, t, u):
        dxdt = np.zeros(self.state_dimension)
        dxdt[0] = x[0] + self.dt * x[1]
        dxdt[1] = x[1] + self.dt * (-u[0] * x[1] - u[1] * x[0] - u[2] * x[0] ** 3)
        return dxdt

    def G(self, x, t, u):
        return x

    def Ac(self, t):
        Ac = np.zeros([self.state_dimension, self.state_dimension])
        Ac[0, 1] = 1
        Ac[1, 0] = -self.nominal_u_interpolated(t)[1] - 3 * self.nominal_u_interpolated(t)[2] * \
                   self.nominal_x_interpolated(t)[0] ** 2
        Ac[1, 1] = -self.nominal_u_interpolated(t)[0]
        return Ac

    def dPhi(self, Phi, t):
        return np.matmul(self.Ac(t), Phi.reshape(self.state_dimension, self.state_dimension)).reshape(
            self.state_dimension ** 2)

    def A(self, tk):
        A = odeint(self.dPhi, np.eye(self.state_dimension).reshape(self.state_dimension ** 2),
                   np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return A[-1, :].reshape(self.state_dimension, self.state_dimension)

    def Bc(self, t):
        Bc = np.zeros([self.state_dimension, self.input_dimension])
        Bc[1, 0] = -self.nominal_x_interpolated(t)[1]
        Bc[1, 1] = -self.nominal_x_interpolated(t)[0]
        Bc[1, 2] = -self.nominal_x_interpolated(t)[0] ** 3
        return Bc

    def dPsi(self, Psi, t):
        return np.matmul(self.Ac(t), Psi.reshape(self.state_dimension, self.state_dimension)).reshape(
            self.state_dimension ** 2) + np.eye(self.state_dimension).reshape(self.state_dimension ** 2)

    def B(self, tk):
        B = odeint(self.dPsi, np.zeros([self.state_dimension, self.state_dimension]).reshape(self.state_dimension ** 2),
                   np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return np.matmul(B[-1, :].reshape(self.state_dimension, self.state_dimension), self.Bc(tk))

    def C(self, tk):
        C = np.eye(self.state_dimension)
        return C

    def D(self, tk):
        D = np.zeros([self.output_dimension, self.input_dimension])
        return D