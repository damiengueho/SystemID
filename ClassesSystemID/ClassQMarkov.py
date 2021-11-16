"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


from SystemIDAlgorithms.QMarkovCover import qMarkovCover


class ERA:
    def __init__(self, markov_parameters, covariance_parameters, Q, state_dimension, **kwargs):
        self.A, self.B, self.C, self.D = qMarkovCover(markov_parameters, covariance_parameters, Q, state_dimension, **kwargs)

