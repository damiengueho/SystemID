"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""


from systemID.SystemIDAlgorithms.BilinearSystemID import bilinearSystemID, bilinearSystemIDFromInitialConditionResponse



class BilinearSystemID:
    def __init__(self, experiments_1, experiments_2, state_dimension, dt, **kwargs):
        self.A, self.N, self.B, self.C, self.D, self.Sigma = bilinearSystemID(experiments_1, experiments_2, state_dimension, dt, **kwargs)



class BilinearSystemIDIC:
    def __init__(self, experiments_1, experiments_2, state_dimension, dt, **kwargs):
        self.A, self.N, self.B, self.C, self.D, self.X0, self.Sigma = bilinearSystemIDFromInitialConditionResponse(experiments_1, experiments_2, state_dimension, dt, **kwargs)