"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


import numpy as np

from systemID.SystemIDAlgorithms.EigenSystemRealizationAlgorithmFromInitialConditionResponse import eigenSystemRealizationAlgorithmFromInitialConditionResponse

def dynamicModeDecompositionAlgorithm(output_signals, state_dimension, input_dimension):

    A, B, C, D, x0, H0, H1, R, Sigma, St, Rn, Sigman, Snt, Op, Rq = eigenSystemRealizationAlgorithmFromInitialConditionResponse(output_signals, state_dimension, input_dimension, p=1)

    def F(tk):
        return np.matmul(LA.sqrtm(Sigman), np.matmul(A, LA.inv(LA.sqrtm(Sigman))))

    return A, B, C, D, F, x0, H0, H1, R, Sigma, St, Rn, Sigman, Snt, Op, Rq
