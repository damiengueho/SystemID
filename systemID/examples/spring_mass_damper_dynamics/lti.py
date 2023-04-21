# Import all necessary packages
import numpy as np
import scipy.interpolate as interp
import scipy.linalg as LA

import matplotlib.pyplot as plt
from matplotlib import rc

import systemID as sysID

# Define parameters
m = 1
c = 0.1
k = 1

state_dimension = 2
input_dimension = 1
output_dimension = 2

frequency = 10

# Create dynamics, measurement equation and associated linear system
def F(x, t, u):
    return np.array([x[1], -c/m * x[1] - k/m * x[0] + u(t)])
def G(x, t, u):
    return x

(Ad, Bd) = sysID.continuous_to_discrete_matrices(1/frequency, np.array([[0, 1], [-k/m, -c/m]]), Bc=np.array([[0], [1]]))
def A(t):
    return Ad
def B(t):
    return Bd
def C(t):
    return np.eye(state_dimension)
def D(t):
    return np.array([[0], [0]])

x0 = np.zeros(state_dimension)
true_system = sysID.continuous_nonlinear_model(x0, F, G=G, input_dimension=1)
true_system_lin = sysID.discrete_linear_model(frequency, x0, A, B=B, C=C, D=D)

# Training
total_time_training = 5
number_steps_training = int(total_time_training * frequency + 1)
tspan_training = np.linspace(0, total_time_training, number_steps_training)

zero_order_hold_data = np.random.randn(number_steps_training)
input_training_continuous = sysID.continuous_signal(u=interp.interp1d(tspan_training, zero_order_hold_data, kind='zero', bounds_error=False, fill_value=zero_order_hold_data[-1]))
input_training_discrete = sysID.discrete_signal(frequency=frequency, data=zero_order_hold_data)
output_training = sysID.propagate(input_training_continuous, true_system, tspan=tspan_training)[0]
output_training_lin = sysID.propagate(input_training_discrete, true_system_lin)[0]

err = output_training.data - output_training_lin.data




# Identification
observer_order = 4
okid_ = sysID.okid_with_observer([input_training_discrete], [output_training], observer_order=observer_order, number_of_parameters=50)

p = 10
q = p
era_ = sysID.era(okid_.markov_parameters, state_dimension=state_dimension, p=p, q=q)

x0_id = np.zeros(state_dimension)
identified_system = sysID.discrete_linear_model(frequency, x0_id, era_.A, B=era_.B, C=era_.C, D=era_.D)

# Testing
total_time_testing = 20
number_steps_testing = int(total_time_testing * frequency + 1)
tspan_testing = np.linspace(0, total_time_testing, number_steps_testing)

omega = 3
def u_testing(t):
    return np.sin(omega * t)
# input_testing_continuous = sysID.continuous_signal(u=interp.interp1d(tspan_testing, u_testing(tspan_testing), kind='zero', bounds_error=False, fill_value=u_testing(tspan_testing)[-1]))
input_testing_discrete = sysID.discrete_signal(frequency=frequency, data=u_testing(tspan_testing))
# output_testing_true = sysID.propagate(input_testing_continuous, true_system, tspan=tspan_testing)[0]
output_testing_true = sysID.propagate(input_testing_discrete, true_system_lin, tspan=tspan_testing)[0]
output_testing_identified = sysID.propagate(input_testing_discrete, identified_system)[0]

# Plotting
plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.serif": ["Computer Modern Roman"]})
rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

fig = plt.figure(num=1, figsize=[8, 5])

ax = fig.add_subplot(2, 1, 1)
ax.plot(tspan_testing, output_testing_true.data[0, :], color=(11/255, 36/255, 251/255), label=r'True')
ax.plot(tspan_testing, output_testing_identified.data[0, :], '--', color=(221/255, 10/255, 22/255), label=r'Identified')
plt.ylabel(r'Position $x$', fontsize=15)
plt.title(r'Comparison True vs. Identified', fontsize=18)
ax.legend(loc='upper center', bbox_to_anchor=(1.18, 1.05), ncol=1, fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

ax = fig.add_subplot(2, 1, 2)
ax.plot(tspan_testing, output_testing_true.data[1, :], color=(11/255, 36/255, 251/255))
ax.plot(tspan_testing, output_testing_identified.data[1, :], '--', color=(221/255, 10/255, 22/255))
plt.xlabel(r'Time [sec]', fontsize=15)
plt.ylabel(r'Velocity $\dot{x}$', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.tight_layout()
plt.show()


fig = plt.figure(num=2, figsize=[8, 3])

ax = fig.add_subplot(1, 1, 1)
ax.plot(tspan_testing, LA.norm(output_testing_true.data - output_testing_identified.data, axis=0), color=(145/255, 145/255, 145/255), label=r'Error')
plt.ylabel(r'Norm of 2-norm error', fontsize=15)
plt.xlabel(r'Time [sec]', fontsize=15)
plt.title(r'Error True vs. Identified', fontsize=18)
ax.legend(loc='upper center', bbox_to_anchor=(1.18, 1.05), ncol=1, fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.tight_layout()
plt.show()


fig = plt.figure(num=2, figsize=[8, 3])

ax = fig.add_subplot(1, 1, 1)
ax.plot(tspan_testing, LA.norm(output_testing_true.data - output_testing_identified.data, axis=0), color=(145/255, 145/255, 145/255), label=r'Error')
plt.ylabel(r'Norm of 2-norm error', fontsize=15)
plt.xlabel(r'Time [sec]', fontsize=15)
plt.title(r'Error True vs. Identified', fontsize=18)
ax.legend(loc='upper center', bbox_to_anchor=(1.18, 1.05), ncol=1, fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.tight_layout()
plt.show()