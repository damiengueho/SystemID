"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""




def findPreviousPowerOf2(n):
    """
        Purpose:
            Compute the previous power of 2 of an integer :math:`n`.

        Parameters:
            - **n** (``int``): the integer.

        Returns:
            - **m** (``int``): previous power of 2 of **n**.

        Description:
            For any integer :math:`n`, :math:`\\exists ! \ d \\in \\mathbb{N}` such that

            .. math::

                2^{d-1} \leq n < 2^d.

            The program returns :math:`2^{d-1}`.

        See Also:
            - :py:mod:`~SystemIDAlgorithms.GetPowerOf2.findNextPowerOf2`

    """

    n = n - 1

    while n & n - 1:
        n = n & n - 1

    return n


def findNextPowerOf2(n):
    """
        Purpose:
            Compute the next power of 2 of an integer :math:`n`.

        Parameters:
            - **n** (``int``): the integer.

        Returns:
            - **m** (``int``): next power of 2 of **n**.

        Description:
            For any integer :math:`n`, :math:`\\exists ! \ d \\in \\mathbb{N}` such that

            .. math::

                2^{d-1} \leq n < 2^d.

            The program returns :math:`2^{d}`.

        See Also:
            - :py:mod:`~SystemIDAlgorithms.GetPowerOf2.findPreviousPowerOf2`

    """

    n = n - 1

    while n & n - 1:
        n = n & n - 1

    return n << 1



