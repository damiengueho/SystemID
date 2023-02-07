import numpy as np
import scipy.linalg as LA


def ar_identification(output_signals, na, nb):
    """
    na > len(output_signals[i])
    nb > len(input_signals[i])
    include capability to identify parameters from any section of the signals?
    nb >= 1 -> otherwise use ar model
    """

    number_signals = len(input_signals)
    output_dimension = output_signals[0].dimension
    input_dimension = input_signals[0].dimension
    number_steps = output_signals[0].number_steps
    n = max(na, nb - 1)

    # Matrix y
    Y = output_signals[0].data
    for k in range(1, number_signals):
        Y = np.concatenate((Y, output_signals[k].data), axis=1)

    # Matrix U
    U = np.zeros([(input_dimension * (nb - 1) + output_dimension * na) + input_dimension, number_steps * number_signals])
    for k in range(number_signals):
        u = input_signals[k].data
        y = output_signals[k].data
        U[0 * input_dimension:(0 + 1) * input_dimension, k * number_steps:(k+1) * number_steps] = u
        cta = 0
        ctb = 1
        for i in range(n):
            if cta < na:
                if ctb < nb:
                    U[i * (input_dimension + output_dimension) + input_dimension:(i + 1) * (input_dimension + output_dimension) + input_dimension, k * number_steps + i + 1:(k+1) * number_steps] = np.concatenate((y[:, 0:number_steps - i - 1], u[:, 0:number_steps - i - 1]), axis=0)
                    cta += 1
                    ctb += 1
                else:
                    U[(nb - 1) * input_dimension + i * output_dimension + input_dimension:(nb - 1) * input_dimension + (i + 1) * output_dimension + input_dimension, k * number_steps + i + 1:(k+1) * number_steps] = y[:, 0:number_steps - i - 1]
                    cta += 1
            else:
                if ctb < nb:
                    U[i * input_dimension + na * output_dimension + input_dimension:(i + 1) * input_dimension + na * output_dimension + input_dimension, k * number_steps + i + 1:(k+1) * number_steps] = u[:,0:number_steps - i - 1]
                    ctb += 1

    # Find parameters
    P = np.matmul(Y, LA.pinv(U))
    print('ARX model identification error on coefficients =', LA.norm(Y - np.matmul(P, U)))

    # Extract parameters
    a = []
    b = [P[:, 0:input_dimension]]
    cta = 0
    ctb = 1
    for i in range(n):
        if cta < na:
            if ctb < nb:
                a.append(P[:, input_dimension + i * (input_dimension + output_dimension):input_dimension + i * input_dimension + (i + 1) * output_dimension])
                b.append(P[:, input_dimension + i * (input_dimension + output_dimension) + output_dimension:input_dimension + (i + 1) * (input_dimension + output_dimension)])
                cta += 1
                ctb += 1
            else:
                a.append(P[:, input_dimension + (nb - 1) * input_dimension + i * output_dimension:input_dimension + (nb - 1) * input_dimension + (i + 1) * output_dimension])
                cta += 1
        else:
            if ctb < nb:
                b.append(P[:, input_dimension + i * input_dimension + na * output_dimension:input_dimension + (i + 1) * input_dimension + na * output_dimension])
                ctb += 1

    return a, b, P, U, Y






