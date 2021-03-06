"""
Author: Damien GUEHO
Copyright: Copyright (C) 2022 Damien GUEHO
License: Public Domain
Version: 23
Date: April 2022
Python: 3.7.7
"""


from systemID.SystemIDAlgorithms.QMarkovCover import qMarkovCover


class QMarkov:
    def __init__(self, markov_parameters, covariance_parameters, Q, state_dimension, **kwargs):
        self.A, self.B, self.C, self.D = qMarkovCover(markov_parameters, covariance_parameters, Q, state_dimension, **kwargs)

