import numpy as np
from systemID.signals.discrete import discrete_signal


class ar_model:
    def __init__(self, coefficients_ar):
        self.coefficients_ar = coefficients_ar


def ar_model_propagation(ar_model, number_steps, initial_condition, **kwargs):
    coefficients_ar = ar_model.coefficients_ar
    na = len(coefficients_ar)
    output_dimension = coefficients_ar[0].shape[0]
    data = np.zeros([output_dimension, number_steps])

    y0 = initial_condition
    data[:, 0] = y0

    for i in range(1, number_steps):
        for j in range(min(i, na)):
            data[:, i] += np.matmul(coefficients_ar[j], data[:, i - 1 - j])

    frequency = kwargs.get('frequency', 1)
    output_signal = discrete_signal(data=data, frequency=frequency)
    return output_signal
