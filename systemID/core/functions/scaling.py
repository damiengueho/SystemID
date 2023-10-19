"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 25
"""


import numpy

def scale_data_to_bounds(raw_data: numpy.ndarray,
                         a: int = -1,
                         b: int = 1):

    # Calculate the maximum and minimum values along each dimension
    max_vals = numpy.max(raw_data, axis=(1, 2), keepdims=True)
    min_vals = numpy.min(raw_data, axis=(1, 2), keepdims=True)

    # Scale the signals to the desired range [a, b]
    scaled_data = a + (b - a) * (raw_data - min_vals) / (max_vals - min_vals)

    return scaled_data


def unscale_data_from_bounds(scaled_data: numpy.ndarray,
                             a: int = -1,
                             b: int = 1):

    # Calculate the maximum and minimum values along each dimension
    max_vals = numpy.max(scaled_data, axis=(1, 2), keepdims=True)
    min_vals = numpy.min(scaled_data, axis=(1, 2), keepdims=True)

    # Scale the signals to the desired range [a, b]
    raw_data = ((scaled_data - a) / (b - a)) * (max_vals - min_vals) + min_vals

    return raw_data


def scale_data_to_normal_distribution(raw_data: numpy.ndarray,
                                      m: int = 0,
                                      s: int = 1):

    # Calculate the mean and standard deviation along each dimension
    mean_vals = numpy.mean(raw_data, axis=(1, 2), keepdims=True)
    std_vals = numpy.std(raw_data, axis=(1, 2), keepdims=True)

    # Scale the signals to have a normal distribution with mean m and standard deviation s
    scaled_data = m + s * ((raw_data - mean_vals) / std_vals)

    return scaled_data


def unscale_data_from_normal_distribution(scaled_data: numpy.ndarray,
                                          m: int = 0,
                                          s: int = 1):

    # Calculate the mean and standard deviation along each dimension of the scaled data
    mean_vals = numpy.mean(scaled_data, axis=(1, 2), keepdims=True)
    std_vals = numpy.std(scaled_data, axis=(1, 2), keepdims=True)

    # Unscaled the signals from the normal distribution with mean m and standard deviation s to the original range
    unscaled_data = ((scaled_data - m) / s) * std_vals + mean_vals

    return unscaled_data
