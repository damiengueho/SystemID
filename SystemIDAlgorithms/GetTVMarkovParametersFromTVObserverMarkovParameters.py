"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 15
Date: August 2021
Python: 3.7.7
"""



import numpy as np
import scipy.linalg as LA


def getTVMarkovParametersFromTVObserverMarkovParameters(D, hki_observer1, hki_observer2, p, q):

    # Dimensions and number of steps
    output_dimension, input_dimension, number_steps = D.shape

    # Create empty hki matrix
    hki = np.zeros([(number_steps - 1) * output_dimension, (number_steps - 1) * input_dimension])

    for k in range(p+q-1-1, number_steps-1):

        # Build matrix h2
        h2 = np.eye((p+q-1) * output_dimension)
        for i in range(p+q-1-1):
            h2[i*output_dimension:(i+1)*output_dimension, (i + 1)*output_dimension:] = hki_observer2[(k - i) * output_dimension:(k + 1 - i) * output_dimension, 0:(p + q - 1 - 1 - i) * output_dimension]

        # Build matrix r
        r = np.zeros([(p+q-1) * output_dimension, (p+q-1) * input_dimension])
        for i in range(p+q-1):
            for j in range(i, p+q-1):
                r[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension] = hki_observer1[(k - i) * output_dimension:(k + 1 - i) * output_dimension, (j - i) * input_dimension:(j - i + 1) * input_dimension] \
                                                                                                                    - np.matmul(hki_observer2[(k - i) * output_dimension:(k + 1 - i) * output_dimension, (j - i) * output_dimension:(j - i + 1) * output_dimension], D[:, :, k - j])

        # Calculate Markov Parameters
        h = np.matmul(LA.inv(h2), r)
        #print('k =', k)
        #print('h2', h2)
        #u, s, v = LA.svd(h2)
        #print('s', s)
        #print('h', h)
        # print('norm(h) =', LA.norm(h))
        # print('norm(h2th2)', LA.norm(np.matmul(h2, LA.inv(h2)) - np.eye((p+q-1) * output_dimension)))
        # print('r', r)
        # print('norm(r)', LA.norm(r))

        # Populate hki
        for i in range(p+q-1):
            for j in range(i, p+q-1):
                hki[(k - i) * output_dimension:(k + 1 - i) * output_dimension, (j - i) * input_dimension:(j - i + 1) * input_dimension] = h[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension]

    return hki

