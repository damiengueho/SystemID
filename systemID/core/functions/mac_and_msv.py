"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""


import numpy
import scipy


def mac_and_msv(A: numpy.ndarray, B: numpy.ndarray, C: numpy.ndarray, Rq: numpy.ndarray, q: int):
    """
        Purpose:
            Compute the Modal Amplitude Coherence (MAC) and Mode Singular Value (MSV) of a linear system.


        Parameters:
            - **A** (``numpy.ndarray``): system matrix
            - **B** (``numpy.ndarray``): input influence matrix
            - **C** (``numpy.ndarray``): output influence matrix
            - **R** (``numpy.ndarray``): controllability matrix of rank q
            - **q** (``int``): size of Rq

        Returns:
            - **MAC** (``list``): MAC values.
            - **MSV** (``list``): MSV values.

        Imports:
            - import numpy
            - import scipy

        Description:
            abc


        See Also:
            -
    """

    # Shapes
    n, r = B.shape
    m, _ = C.shape

    # Initialize MAC and MSV
    MAC = numpy.zeros(n)
    MSV = numpy.zeros(n)

    # Transform to modal coordinates
    ev, T = scipy.linalg.eig(A)
    A_m = numpy.diag(ev)
    B_m = numpy.matmul(scipy.linalg.inv(T), B)
    C_m = numpy.matmul(C, T)
    Rq_m = numpy.matmul(scipy.linalg.inv(T), Rq)

    # Controllability-like matrix
    Q = numpy.zeros(Rq.shape, dtype=complex)

    # Compute MAC and MSV
    for k in range(n):
        s_lambda = 0
        for i in range(q):
            Q[k, r*i:r*(i+1)] = A_m[k, k] ** i * B_m[k, :]
            s_lambda = s_lambda + numpy.abs(A_m[k, k]**i)
        MAC[k] = numpy.abs(numpy.dot(Q[k, :], numpy.conj(Rq_m[k, :]))) / numpy.sqrt(numpy.abs(numpy.dot(Q[k, :], numpy.conj(Q[k, :])))*numpy.abs(numpy.dot(Rq_m[k, :], numpy.conj(Rq_m[k, :]))))
        MSV[k] = numpy.sqrt(scipy.linalg.norm(C_m[:, k]) * s_lambda * scipy.linalg.norm(B_m[k, :]))

    return MAC, MSV
