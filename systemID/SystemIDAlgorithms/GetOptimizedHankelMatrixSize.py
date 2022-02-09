"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""



import numpy as np


def getOptimizedHankelMatrixSize(assumed_order, output_dimension, input_dimension):
    """
    Purpose:


    Parameters:
        -

    Returns:
        -

    Imports:
        -

    Description:


    See Also:
        -
    """

    p = int(np.ceil(assumed_order / output_dimension))
    q = int(np.ceil(assumed_order / input_dimension))

    return p, q