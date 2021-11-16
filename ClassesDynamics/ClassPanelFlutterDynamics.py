"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""



import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from ClassesGeneral.ClassSignal import DiscreteSignal


class PanelFlutterDynamics:
    def __init__(self, RT, mu, M, l, **kwargs):
        self.state_dimension = 4
        self.input_dimension = 2
        self.output_dimension = 4
        self.RT = RT
        self.mu = mu
        self.M = M
        self.l = l
        self.tspan = kwargs.get('tspan', np.array([0, 1, 2, 3]))
        self.nominal_x = kwargs.get('nominal_x', DiscreteSignal(self.state_dimension, 3, 1))
        self.nominal_x_interpolated = interp1d(self.tspan, self.nominal_x.data, 'cubic')
        self.nominal_u = kwargs.get('nominal_u', DiscreteSignal(self.input_dimension, 3, 1))
        self.nominal_u_interpolated = interp1d(self.tspan, self.nominal_u.data, 'cubic')
        self.dt = kwargs.get('dt', 0)

    def F(self, x, t, u):
        dxdt = np.zeros(4)
        dxdt[0] = x[2]
        dxdt[1] = x[3]
        dxdt[2] = -np.pi**4*x[0] + 10*np.pi**2*self.RT(t)*x[0] + 8*self.l(t)*x[1]/3 - 5*np.pi**4*x[0]**3/2 - 10*np.pi**4*x[0]*x[1]**2 - np.sqrt(self.l(t)*self.mu(t)/self.M(t))*x[2] + u(t)[0]
        dxdt[3] = -8*self.l(t)*x[0]/3 - 16*np.pi**4*x[1] + 40*np.pi**2*self.RT(t)*x[1] - 10*np.pi**4*x[0]**2*x[1] - 40*np.pi**4*x[1]**3 - np.sqrt(self.l(t)*self.mu(t)/self.M(t))*x[3] + u(t)[1]
        return dxdt

    def G(self, x, t, u):
        return x

    def Ac(self, t):
        Ac = np.zeros([4, 4])
        Ac[0, 2] = 1
        Ac[1, 3] = 1
        Ac[2, 0] = -np.pi**4 + 10*np.pi**2*self.RT(t) - 15*np.pi**4*self.nominal_x_interpolated(t)[0]**3/2 - 10*np.pi**4*self.nominal_x_interpolated(t)[1]**2
        Ac[2, 1] = 8*self.l(t)/3 - 20*np.pi**4*self.nominal_x_interpolated(t)[0]*self.nominal_x_interpolated(t)[1]
        Ac[2, 2] = -np.sqrt(self.l(t)*self.mu(t)/self.M(t))
        Ac[3, 0] = -8*self.l(t)/3 - 20*np.pi**4*self.nominal_x_interpolated(t)[0]*self.nominal_x_interpolated(t)[1]
        Ac[3, 1] = -16*np.pi**4 + 40*np.pi**2*self.RT(t) - 10*np.pi**4*self.nominal_x_interpolated(t)[0]**2 - 120*np.pi**4*self.nominal_x_interpolated(t)[1]**2
        Ac[3, 3] = -np.sqrt(self.l(t)*self.mu(t)/self.M(t))
        return Ac

    def dPhi(self, Phi, t):
        return np.matmul(self.Ac(t), Phi.reshape(4, 4)).reshape(16)

    def A(self, tk):
        A = odeint(self.dPhi, np.eye(4).reshape(16), np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return A[-1, :].reshape(4, 4)

    def Bc(self, t):
        Bc = np.zeros([4, 2])
        Bc[2, 0] = 1
        Bc[3, 1] = 1
        return Bc

    def dPsi(self, Psi, t):
        return np.matmul(self.Ac(t), Psi.reshape(4, 4)).reshape(16) + np.eye(4).reshape(16)

    def B(self, tk):
        B = odeint(self.dPsi, np.zeros([4, 4]).reshape(16), np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return np.matmul(B[-1, :].reshape(4, 4), self.Bc(tk))

    def C(self, tk):
        C = np.eye(4)
        return C

    def D(self, tk):
        D = np.zeros([4, 2])
        return D





########################################################################################################################
# lambda is a 1-dim input
########################################################################################################################


"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""



import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from ClassesGeneral.ClassSignal import DiscreteSignal


class PanelFlutterDynamics2:
    def __init__(self, RT, mu, M, **kwargs):
        self.state_dimension = 4
        self.input_dimension = 1
        self.output_dimension = 4
        self.RT = RT
        self.mu = mu
        self.M = M
        self.tspan = kwargs.get('tspan', np.array([0, 1, 2, 3]))
        self.nominal_x = kwargs.get('nominal_x', DiscreteSignal(self.state_dimension, 'No nominal trajectory', 3, 1))
        self.nominal_x_interpolated = interp1d(self.tspan, self.nominal_x.data, 'cubic')
        self.nominal_u = kwargs.get('nominal_u', DiscreteSignal(self.input_dimension, 'No nominal input', 3, 1))
        self.nominal_u_interpolated = interp1d(self.tspan, self.nominal_u.data, 'cubic')
        self.dt = kwargs.get('dt', 0)

    def F(self, x, t, u):
        dxdt = np.zeros(4)
        dxdt[0] = x[2]
        dxdt[1] = x[3]
        dxdt[2] = -np.pi**4*x[0] + 10*np.pi**2*self.RT(t)*x[0] + 8*u(t)*x[1]/3 - 5*np.pi**4*x[0]**3/2 - 10*np.pi**4*x[0]*x[1]**2 - np.sqrt(u(t)*self.mu(t)/self.M(t))*x[2]
        dxdt[3] = -8*u(t)*x[0]/3 - 16*np.pi**4*x[1] + 40*np.pi**2*self.RT(t)*x[1] - 10*np.pi**4*x[0]**2*x[1] - 40*np.pi**4*x[1]**3 - np.sqrt(u(t)*self.mu(t)/self.M(t))*x[3]
        return dxdt

    def G(self, x, t, u):
        return x

    def Ac(self, t):
        Ac = np.zeros([self.state_dimension, self.state_dimension])
        Ac[0, 2] = 1
        Ac[1, 3] = 1
        Ac[2, 0] = -np.pi**4 + 10*np.pi**2*self.RT(t) - 15*np.pi**4*self.nominal_x_interpolated(t)[0]**3/2 - 10*np.pi**4*self.nominal_x_interpolated(t)[1]**2
        Ac[2, 1] = 8*self.nominal_u_interpolated(t)/3 - 20*np.pi**4*self.nominal_x_interpolated(t)[0]*self.nominal_x_interpolated(t)[1]
        Ac[2, 2] = -np.sqrt(self.nominal_u_interpolated(t)*self.mu(t)/self.M(t))
        Ac[3, 0] = -8*self.nominal_u_interpolated(t)/3 - 20*np.pi**4*self.nominal_x_interpolated(t)[0]*self.nominal_x_interpolated(t)[1]
        Ac[3, 1] = -16*np.pi**4 + 40*np.pi**2*self.RT(t) - 10*np.pi**4*self.nominal_x_interpolated(t)[0]**2 - 120*np.pi**4*self.nominal_x_interpolated(t)[1]**2
        Ac[3, 3] = -np.sqrt(self.nominal_u_interpolated(t)*self.mu(t)/self.M(t))
        return Ac

    def dPhi(self, Phi, t):
        return np.matmul(self.Ac(t), Phi.reshape(4, 4)).reshape(16)

    def A(self, tk):
        A = odeint(self.dPhi, np.eye(4).reshape(16), np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return A[-1, :].reshape(4, 4)

    def Bc(self, t):
        Bc = np.zeros([self.state_dimension, self.input_dimension])
        Bc[2, 0] = 1
        Bc[3, 1] = 1
        return Bc

    def dPsi(self, Psi, t):
        return np.matmul(self.Ac(t), Psi.reshape(4, 4)).reshape(16) + np.eye(4).reshape(16)

    def B(self, tk):
        B = odeint(self.dPsi, np.zeros([4, 4]).reshape(16), np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return np.matmul(B[-1, :].reshape(4, 4), self.Bc(tk))

    def C(self, tk):
        C = np.eye(self.state_dimension)
        return C

    def D(self, tk):
        D = np.zeros([4, self.input_dimension])
        return D





########################################################################################################################
# lambda is a 2-dim input
########################################################################################################################


"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""



import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from ClassesGeneral.ClassSignal import DiscreteSignal


class PanelFlutterDynamics3:
    def __init__(self, RT, **kwargs):
        self.state_dimension = 4
        self.input_dimension = 2
        self.output_dimension = 4
        self.RT = RT
        self.tspan = kwargs.get('tspan', np.array([0, 1, 2, 3]))
        self.nominal_x = kwargs.get('nominal_x', DiscreteSignal(self.state_dimension, 3, 1))
        self.nominal_x_interpolated = interp1d(self.tspan, self.nominal_x.data, 'cubic', bounds_error=False, fill_value="extrapolate")
        self.nominal_u = kwargs.get('nominal_u', DiscreteSignal(self.input_dimension, 3, 1))
        self.nominal_u_interpolated = interp1d(self.tspan, self.nominal_u.data, 'cubic', bounds_error=False, fill_value="extrapolate")
        self.dt = kwargs.get('dt', 0)

    def F(self, x, t, u):
        dxdt = np.zeros(self.state_dimension)
        dxdt[0] = x[2]
        dxdt[1] = x[3]
        dxdt[2] = -np.pi**4*x[0] + 10*np.pi**2*self.RT(t)*x[0] + 8*u(t)[0]*x[1]/3 - 5*np.pi**4*x[0]**3/2 - 10*np.pi**4*x[0]*x[1]**2 - u(t)[1]*x[2]
        dxdt[3] = -8*u(t)[0]*x[0]/3 - 16*np.pi**4*x[1] + 40*np.pi**2*self.RT(t)*x[1] - 10*np.pi**4*x[0]**2*x[1] - 40*np.pi**4*x[1]**3 - u(t)[1]*x[3]
        return dxdt

    def G(self, x, t, u):
        return x

    def Ac(self, t):
        Ac = np.zeros([self.state_dimension, self.state_dimension])
        Ac[0, 2] = 1
        Ac[1, 3] = 1
        Ac[2, 0] = -np.pi**4 + 10*np.pi**2*self.RT(t) - 15*np.pi**4*self.nominal_x_interpolated(t)[0]**3/2 - 10*np.pi**4*self.nominal_x_interpolated(t)[1]**2
        Ac[2, 1] = 8*self.nominal_u_interpolated(t)[0]/3 - 20*np.pi**4*self.nominal_x_interpolated(t)[0]*self.nominal_x_interpolated(t)[1]
        Ac[2, 2] = -self.nominal_u_interpolated(t)[1]
        Ac[3, 0] = -8*self.nominal_u_interpolated(t)[0]/3 - 20*np.pi**4*self.nominal_x_interpolated(t)[0]*self.nominal_x_interpolated(t)[1]
        Ac[3, 1] = -16*np.pi**4 + 40*np.pi**2*self.RT(t) - 10*np.pi**4*self.nominal_x_interpolated(t)[0]**2 - 120*np.pi**4*self.nominal_x_interpolated(t)[1]**2
        Ac[3, 3] = -self.nominal_u_interpolated(t)[1]
        return Ac

    def dPhi(self, Phi, t):
        return np.matmul(self.Ac(t), Phi.reshape(self.state_dimension, self.state_dimension)).reshape(self.state_dimension ** 2)

    def A(self, tk):
        A = odeint(self.dPhi, np.eye(self.state_dimension).reshape(self.state_dimension ** 2), np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return A[-1, :].reshape(self.state_dimension, self.state_dimension)

    def Bc(self, t):
        Bc = np.zeros([self.state_dimension, self.input_dimension])
        Bc[2, 0] = 8*self.nominal_x_interpolated(t)[1]/3
        Bc[2, 1] = -self.nominal_x_interpolated(t)[2]
        Bc[3, 0] = -8*self.nominal_x_interpolated(t)[0]/3
        Bc[3, 1] = -self.nominal_x_interpolated(t)[3]
        return Bc

    def dPsi(self, Psi, t):
        return np.matmul(self.Ac(t), Psi.reshape(self.state_dimension, self.state_dimension)).reshape(self.state_dimension ** 2) + np.eye(self.state_dimension).reshape(self.state_dimension ** 2)

    def B(self, tk):
        B = odeint(self.dPsi, np.zeros([self.state_dimension, self.state_dimension]).reshape(self.state_dimension ** 2), np.array([tk, tk + self.dt]), rtol=1e-13, atol=1e-13)
        return np.matmul(B[-1, :].reshape(self.state_dimension, self.state_dimension), self.Bc(tk))

    def C(self, tk):
        C = np.eye(self.state_dimension)
        return C

    def D(self, tk):
        D = np.zeros([self.output_dimension, self.input_dimension])
        return D