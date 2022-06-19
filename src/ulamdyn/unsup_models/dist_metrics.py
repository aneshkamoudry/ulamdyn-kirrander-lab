"""This module provides a set of methods to calculate distances between pairs of molecular geometries."""
# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: 04/06/2022
import rmsd
import numpy as np

from sklearn.metrics import pairwise_distances

__all__ = ["calc_rmsd"]


def calc_rmsd(x1, x2):
    """Calculate the root-mean square deviation between two aligned geometries.

    :param x1: Matrix of Cartesian coordinates for geometry 1.
    :type x1: numpy.ndarray
    :param x2: Matrix of Cartesian coordinates for geometry 2.
    :type x2: numpy.ndarray
    :return: RMSD value computed for the two input geometries.
    :rtype: float
    """
    x1 = x1.reshape(-1, 3)
    x2 = x2.reshape(-1, 3)
    rmsd_val = rmsd.rmsd(x1, x2)
    return rmsd_val
