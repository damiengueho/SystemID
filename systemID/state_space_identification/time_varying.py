"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""

from systemID.functions.time_varying_eigensystem_realization_algorithm import time_varying_eigensystem_realization_algorithm
from systemID.functions.time_varying_eigensystem_realization_algorithm_from_initial_condition_response import time_varying_eigensystem_realization_algorithm_from_initial_condition_response
# from systemID.SystemIDAlgorithms.TimeVaryingEigenSystemRealizationAlgorithmWithDataCorrelation import timeVaryingEigenSystemRealizationAlgorithmWithDataCorrelation
# from systemID.SystemIDAlgorithms.TimeVaryingEigenSystemRealizationAlgorithmWithDataCorrelationFromInitialConditionResponse import timeVaryingEigenSystemRealizationAlgorithmWithDataCorrelationFromInitialConditionResponse


class tvera:
    def __init__(self, hki, D, frequency, state_dimension, p, q, **kwargs):
        self.A, self.B, self.C, self.D, self.Ok, self.Ok1, self.Sigma, self.X0, self.A_id, self.B_id, self.C_id, self.D_id = time_varying_eigensystem_realization_algorithm(hki, D, frequency, state_dimension, p, q, **kwargs)


class tvera_ic:
    def __init__(self, output_signals, state_dimension, p, **kwargs):
        self.A, self.C, self.Ok, self.Ok1, self.Sigma, self.X0, self.A_id, self.C_id, self.MAC, self.MSV, self.Y = time_varying_eigensystem_realization_algorithm_from_initial_condition_response(output_signals, state_dimension, p, **kwargs)



# class TVERADC:
#     def __init__(self, free_decay_experiments, hki, D, full_experiment, state_dimension, **kwargs):
#         self.A, self.B, self.C, self.D, self.x0, self.xq, self.Ok, self.Ok1, self.sigma, self.Hpnt, self.Rkt, self.Hkxzt, self.Hkxz, self.Rk = timeVaryingEigenSystemRealizationAlgorithmWithDataCorrelation(free_decay_experiments, hki, D, full_experiment, state_dimension, **kwargs)
#
#
#
# class TVERADCFromInitialConditionResponse:
#     def __init__(self, free_decay_experiments, state_dimension, **kwargs):
#         self.A, self.B, self.C, self.D, self.X0, self.Ok, self.Ok1, self.sigma, self.Hpnt, self.Hkxzt, self.Rkt = timeVaryingEigenSystemRealizationAlgorithmWithDataCorrelationFromInitialConditionResponse(free_decay_experiments, state_dimension, **kwargs)

