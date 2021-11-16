"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""



# from SparseApproximation import sparseApproximation
# from SparseApproximation2 import sparseApproximation2
# from SparseApproximation2_TwoBodyProblem_v2 import sparseApproximation2_TwoBodyProblemv2
# from SparseApproximation2_TwoBodyProblem_polar import sparseApproximation2_TwoBodyProblem_polar

from SparseIDAlgorithms.SparseApproximation2ndOrder import sparseApproximation2ndOrder
from SparseIDAlgorithms.SparseApproximation1stOrder import sparseApproximation1stOrder


# class SparseApproximation:
#     def __init__(self, signal, input_signal, order, max_order, post_treatment, l, alpha, delta, max_iterations):
#         self.interp_x, self.interp_u, self.index, self.THETA_LS, self.THETA_SPARSE, self.ZX, self.ZU, self.Psix, self.xLS, self.xSPARSE = sparseApproximation(signal, input_signal, order, max_order, post_treatment, l, alpha, delta, max_iterations)
#
# class SparseApproximation2:
#     def __init__(self, signal, dx0, input_signal, order, max_order, post_treatment, l1, l2, alpha, delta, max_iterations):
#         self.interp_x, self.interp_u, self.THETA_LS, self.THETA_SPARSE, self.ZX, self.ZV, self.ZL, self.ZL_dot, self.ZU, self.ZU_dot, self.Psi, self.Psi_dot, self.xLS, self.xSPARSE = sparseApproximation2(signal, dx0, input_signal, order, max_order, post_treatment, l1, l2, alpha, delta, max_iterations)
#
# class SparseApproximation2_TwoBodyProblemv2:
#     def __init__(self, signal, dx0, input_signal, order, max_order, post_treatment, l1, l2, alpha, delta, max_iterations, TU, shift):
#         self.interp_x, self.interp_u, self.index, self.THETA_LS, self.THETA_SPARSE, self.ZX, self.ZV, self.ZL, self.ZL_dot, self.ZU, self.ZU_dot, self.PSI, self.LS_signals, self.Sparse_signals, self.C = sparseApproximation2_TwoBodyProblemv2(signal, dx0, input_signal, order, max_order, post_treatment, l1, l2, alpha, delta, max_iterations, TU, shift)
#
# class SparseApproximation2_TwoBodyProblem_polar:
#     def __init__(self, signal, dx0, input_signal, order, max_order, post_treatment, l1, l2, alpha, delta, max_iterations, TU):
#         self.interp_x, self.interp_u, self.index, self.THETA_LS, self.THETA_SPARSE, self.ZX, self.ZV, self.ZL, self.ZL_dot, self.ZU, self.ZU_dot, self.Psi, self.Psi_dot, self.xLS, self.xSPARSE = sparseApproximation2_TwoBodyProblem_polar(signal, dx0, input_signal, order, max_order, post_treatment, l1, l2, alpha, delta, max_iterations, TU)

class SparseApproximation2ndOrder:
    def __init__(self, signals, dx0s, input_signals, order, max_order, post_treatment, l1, l2, alpha, delta, epsilon, max_iterations, shift):
        self.interp_data, self.interp_inputs, self.index, self.THETA_LS, self.THETA_SPARSE, self.Y1, self.Y2, self.dY2, self.U, self.dU, self.PHI, self.dPHI, self.C, self.LS_signals, self.Sparse_signals = sparseApproximation2ndOrder(signals, dx0s, input_signals, order, max_order, post_treatment, l1, l2, alpha, delta, epsilon, max_iterations, shift)


class SparseApproximation1stOrder:
    def __init__(self, signals, input_signals, x0s_testing, input_signals_testing, order, max_order, post_treatment, l1, alpha, delta, epsilon, max_iterations, shift):
        self.interp_data, self.interp_inputs, self.index, self.THETA_LS, self.THETA_SPARSE, self.Y1, self.U, self.PHI, self.C, self.LS_signals, self.Sparse_signals, self.LS_signals_testing, self.Sparse_signals_testing = sparseApproximation1stOrder(signals, input_signals, x0s_testing, input_signals_testing, order, max_order, post_treatment, l1, alpha, delta, epsilon, max_iterations, shift)
