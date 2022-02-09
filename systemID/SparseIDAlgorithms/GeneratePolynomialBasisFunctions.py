"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""




def generatePolynomialBasisFunctions(dimension, index):

    index_length, _ = index.shape
    basis_functions = []

    def make_Phix(I):
        def Phix(x):
            temp = 1
            for k in range(dimension):
                temp = temp*x[k]**I[k]
            return temp
        return Phix

    for i in range(index_length):
        basis_functions.append(make_Phix(index[i, 0:dimension]))

    return basis_functions