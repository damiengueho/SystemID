"""
Author: Damien GUEHO
Copyright: Copyright (C) 2022 Damien GUEHO
License: Public Domain
Version: 23
Date: April 2022
Python: 3.7.7
"""


from systemID.SystemIDAlgorithms.BilinearSystemID import bilinearSystemID, bilinearSystemIDFromInitialConditionResponse, bilinearSystemIDFromInitialConditionResponseConstantInput, bilinearSystemIDFromInitialConditionResponseConstantInput2



class BilinearSystemID:
    def __init__(self, experiments_1, experiments_2, state_dimension, dt, **kwargs):
        self.A, self.N, self.B, self.C, self.D, self.Sigma = bilinearSystemID(experiments_1, experiments_2, state_dimension, dt, **kwargs)



class BilinearSystemIDIC:
    def __init__(self, experiments_1, experiments_2, state_dimension, dt, **kwargs):
        self.A, self.N, self.B, self.C, self.D, self.X0, self.Sigma = bilinearSystemIDFromInitialConditionResponse(experiments_1, experiments_2, state_dimension, dt, **kwargs)



class BilinearSystemIDICConstantInput:
    def __init__(self, experiments_1, experiments_2, state_dimension, dt, ind, l, **kwargs):
        self.A, self.N, self.B, self.C, self.D, self.X0, self.Sigma, self.LAA, self.V0_1N2, self.CR, self.Op, self.exp1, self.exp2 = bilinearSystemIDFromInitialConditionResponseConstantInput(experiments_1, experiments_2, state_dimension, dt, ind, l, **kwargs)



class BilinearSystemIDICConstantInput2:
    def __init__(self, experiments_1, state_dimension, dt, **kwargs):
        self.A, self.N, self.B, self.C, self.D, self.X0, self.Sigma, self.LAA, self.V0_1N2, self.CR, self.Op = bilinearSystemIDFromInitialConditionResponseConstantInput2(experiments_1, state_dimension, dt, **kwargs)
