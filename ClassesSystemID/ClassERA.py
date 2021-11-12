"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 19
Date: November 2021
Python: 3.7.7
"""


from SystemIDAlgorithms.EigenSystemRealizationAlgorithm import eigenSystemRealizationAlgorithm
from SystemIDAlgorithms.EigenSystemRealizationAlgorithmFromInitialConditionResponse import eigenSystemRealizationAlgorithmFromInitialConditionResponse
from SystemIDAlgorithms.EigenSystemRealizationAlgorithmWithDataCorrelationFromInitialConditionResponse import eigenSystemRealizationAlgorithmWithDataCorrelationFromInitialConditionResponse
from SystemIDAlgorithms.EigenSystemRealizationAlgorithmWithDataCorrelation import eigenSystemRealizationAlgorithmWithDataCorrelation
from SystemIDAlgorithms.DynamicModeDecompositionAlgorithm import dynamicModeDecompositionAlgorithm
from SystemIDAlgorithms.TimeVaryingEigenSystemRealizationAlgorithm import timeVaryingEigenSystemRealizationAlgorithm
from SystemIDAlgorithms.TimeVaryingEigenSystemRealizationAlgorithmFromInitialConditionResponse import timeVaryingEigenSystemRealizationAlgorithmFromInitialConditionResponse
from SystemIDAlgorithms.TimeVaryingEigenSystemRealizationAlgorithmWithDataCorrelation import timeVaryingEigenSystemRealizationAlgorithmWithDataCorrelation
from SystemIDAlgorithms.TimeVaryingEigenSystemRealizationAlgorithmWithDataCorrelationFromInitialConditionResponse import timeVaryingEigenSystemRealizationAlgorithmWithDataCorrelationFromInitialConditionResponse


class ERA:
    def __init__(self, markov_parameters, state_dimension, **kwargs):
        self.A, self.B, self.C, self.D, self.H0, self.H1, self.R, self.Sigma, self.St, self.Rn, self.Sigman, self.Snt, self.Op, self.Rq, self.MAC, self.MSV = eigenSystemRealizationAlgorithm(markov_parameters, state_dimension, **kwargs)



class ERAFromInitialConditionResponse:
    def __init__(self, output_signals, state_dimension, input_dimension, **kwargs):
        self.A, self.B, self.C, self.D, self.X0, self.H0, self.H1, self.R, self.Sigma, self.St, self.Rn, self.Sigman, self.Snt, self.Op, self.Rq, self.MAC, self.MSV = eigenSystemRealizationAlgorithmFromInitialConditionResponse(output_signals, state_dimension, input_dimension, **kwargs)



class ERADC:
    def __init__(self, markov_parameters, state_dimension, **kwargs):
        self.A, self.B, self.C, self.D, self.H0, self.H1, self.R, self.Sigma, self.St, self.Rn, self.Sigman, self.Snt, self.Op, self.Rq, self.MAC, self.MSV = eigenSystemRealizationAlgorithmWithDataCorrelation(markov_parameters, state_dimension, **kwargs)



class ERADCFromInitialConditionResponse:
    def __init__(self, output_signals, true_output_signal, state_dimension, input_dimension, **kwargs):
        self.A, self.B, self.C, self.D, self.X0, self.x0, self.H0, self.H1, self.R, self.Sigma, self.St, self.Rn, self.Sigman, self.Snt, self.Op, self.Rq, self.MAC, self.MSV = eigenSystemRealizationAlgorithmWithDataCorrelationFromInitialConditionResponse(output_signals, true_output_signal, state_dimension, input_dimension, **kwargs)



class DMD:
    def __index__(self, output_signals, state_dimension, input_dimension, **kwargs):
        self.A, self.B, self.C, self.D, self.F, self.x0, self.H0, self.H1, self.R, self.Sigma, self.St, self.Rn, self.Sigman, self.Snt, self.Op, self.Rq = dynamicModeDecompositionAlgorithm(output_signals, state_dimension, input_dimension, **kwargs)



class TVERA:
    def __init__(self, free_decay_experiments, hki, D, state_dimension, p, q, **kwargs):
        self.A, self.B, self.C, self.D, self.xq, self.Ok, self.Ok1, self.sigma = timeVaryingEigenSystemRealizationAlgorithm(free_decay_experiments, hki, D, state_dimension, p, q, **kwargs)



class TVERAFromInitialConditionResponse:
    def __init__(self, free_decay_experiments, state_dimension, p):
        self.A, self.B, self.C, self.D, self.Ok, self.Ok1, self.Sigma, self.X0 = timeVaryingEigenSystemRealizationAlgorithmFromInitialConditionResponse(free_decay_experiments, state_dimension, p)



class TVERADC:
    def __init__(self, free_decay_experiments, hki, D, full_experiment, state_dimension, **kwargs):
        self.A, self.B, self.C, self.D, self.x0, self.xq, self.Ok, self.Ok1, self.sigma, self.Hpnt, self.Rkt, self.Hkxzt, self.Hkxz, self.Rk = timeVaryingEigenSystemRealizationAlgorithmWithDataCorrelation(free_decay_experiments, hki, D, full_experiment, state_dimension, **kwargs)



class TVERADCFromInitialConditionResponse:
    def __init__(self, free_decay_experiments, full_experiment, state_dimension, **kwargs):
        self.A, self.B, self.C, self.D, self.x0, self.Ok, self.Ok1, self.sigma, self.Hpnt, self.Hkxzt, self.Rkt = timeVaryingEigenSystemRealizationAlgorithmWithDataCorrelationFromInitialConditionResponse(free_decay_experiments, full_experiment, state_dimension, **kwargs)
