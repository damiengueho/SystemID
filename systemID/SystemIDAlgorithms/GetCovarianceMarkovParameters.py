"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""


import numpy as np
from scipy import linalg as LA


def getCovarianceMarkovParameters(output_signal, p):
    """
    Purpose:


    Parameters:
        -

    Returns:
        -

    Imports:
        -

    Description:


    See Also:
        -
    """

    covariance_markov_parameters = []

    for i in range(p):
        covariance_markov_parameters.append(np.mean(output_signal.data.T[:, np.newaxis, i:] * output_signal.data.T[:, :-i, np.newaxis], axis=2))

    return covariance_markov_parameters