"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""




def findPreviousPowerOf2(n):

    n = n - 1

    while n & n - 1:
        n = n & n - 1

    return n


def findNextPowerOf2(n):
    n = n - 1

    while n & n - 1:
        n = n & n - 1

    return n << 1
