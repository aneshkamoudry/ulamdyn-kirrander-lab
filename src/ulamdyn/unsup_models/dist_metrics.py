import rmsd
import numpy as np

from sklearn.metrics import pairwise_distances


def calc_rmsd(x1, x2):
    x1 = x1.reshape(-1, 3)
    x2 = x2.reshape(-1, 3)
    rmsd_val = rmsd.rmsd(x1, x2)
    return rmsd_val
