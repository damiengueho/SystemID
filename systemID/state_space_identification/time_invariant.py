"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""


from systemID.functions.eigensystem_realization_algorithm import eigensystem_realization_algorithm
from systemID.functions.eigensystem_realization_algorithm_from_initial_condition_response import eigensystem_realization_algorithm_from_initial_condition_response
from systemID.functions.eigensystem_realization_algorithm_with_data_correlation import eigensystem_realization_algorithm_with_data_correlation
from systemID.functions.eigensystem_realization_algorithm_with_data_correlation_from_initial_condition_response import eigensystem_realization_algorithm_with_data_correlation_from_initial_condition_response

# from systemID.SystemIDAlgorithms.DynamicModeDecompositionAlgorithm import dynamicModeDecompositionAlgorithm


class era:
    def __init__(self, markov_parameters, state_dimension, **kwargs):
        self.A, self.B, self.C, self.D, self.H0, self.H1, self.R, self.Sigma, self.St, self.Rn, self.Sigman, self.Snt, self.Op, self.Rq, self.MAC, self.MSV = eigensystem_realization_algorithm(markov_parameters, state_dimension, **kwargs)


class era_ic:
    def __init__(self, output_signals, state_dimension, **kwargs):
        self.A, self.C, self.X0, self.H0, self.H1, self.R, self.Sigma, self.St, self.Rn, self.Sigman, self.Snt, self.Op, self.Rq, self.MAC, self.MSV = eigensystem_realization_algorithm_from_initial_condition_response(output_signals, state_dimension, **kwargs)


class eradc:
    def __init__(self, markov_parameters, state_dimension, **kwargs):
        self.A, self.B, self.C, self.D, self.H0, self.H1, self.R, self.Sigma, self.St, self.Rn, self.Sigman, self.Snt, self.Op, self.Rq, self.MAC, self.MSV = eigensystem_realization_algorithm_with_data_correlation(markov_parameters, state_dimension, **kwargs)


class eradc_ic:
    def __init__(self, output_signals, state_dimension, **kwargs):
        self.A, self.C, self.X0, self.H0, self.H1, self.R, self.Sigma, self.St, self.Rn, self.Sigman, self.Snt, self.Op, self.Rq, self.MAC, self.MSV = eigensystem_realization_algorithm_with_data_correlation_from_initial_condition_response(output_signals, state_dimension, **kwargs)


# class tiko:
#     def __init__(self,):


# class tiko_ic:
#     def __init__(self, output_signals, state_dimension, **kwargs):
#         output_signals_augmented = augment_signals_with_polynomial_basis_functions(output_signals)



# class DMD:
#     def __index__(self, output_signals, state_dimension, input_dimension, **kwargs):
#         self.A, self.B, self.C, self.D, self.F, self.x0, self.H0, self.H1, self.R, self.Sigma, self.St, self.Rn, self.Sigman, self.Snt, self.Op, self.Rq = dynamicModeDecompositionAlgorithm(output_signals, state_dimension, input_dimension, **kwargs)


