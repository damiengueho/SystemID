import numpy as np
from systemID.signals.discrete import discrete_signal



class arx_model:
    def __init__(self, coefficients_ar, coefficients_x):
        self.coefficients_ar = coefficients_ar
        self.coefficients_x = coefficients_x


def arx_model_propagation(input_signal, arx_model, **kwargs):
    coefficients_ar = arx_model.coefficients_ar
    na = len(coefficients_ar)
    coefficients_x = arx_model.coefficients_x
    nb = len(coefficients_x)
    u = input_signal.data
    number_steps = input_signal.number_steps
    output_dimension = coefficients_x[0].shape[0]
    data = np.zeros([output_dimension, number_steps])

    y0 = kwargs.get('initial_condition', np.matmul(coefficients_x[0], u[:, 0]))
    data[:, 0] = y0
    
    for i in range(1, input_signal.number_steps):
        for j in range(min(i, na)):
            data[:, i] += np.matmul(coefficients_ar[j], data[:, i - 1 - j])
        for j in range(min(i + 1, nb)):
            data[:, i] += np.matmul(coefficients_x[j], u[:, i - j])

    output_signal = discrete_signal(data=data, frequency=input_signal.frequency)
    return output_signal
