"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""



class CustomizedDiscreteTimeInvariantDynamics:
    def __init__(self, A, B, C, D):
        self.state_dimension, _ = A.shape
        self.output_dimension, self.input_dimension = D.shape
        self.Ad = A
        self.Bd = B
        self.Cd = C
        self.Dd = D

    def A(self, tk):
        return self.Ad

    def B(self, tk):
        return self.Bd

    def C(self, tk):
        return self.Cd

    def D(self, tk):
        return self.Dd
