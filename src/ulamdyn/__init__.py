# -*- coding: utf-8 -*-
"""
  ULaMDyn - Unsupervised Learning analysis for Molecular Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ULaMDyn is a python package built on top of sklearn designed to perform
data preprocessing, statistical and unsupervised learning analysis of
(non-adiabatic) molecular dynamics simulations.

ULaMDyn is composed of several Python modules:

+ Data_Loader
+ Data_Writer
+ Statistics
+ Kinetics
+ Descriptors
+ Normal mode analysis
+ Unsup_Models
+ NMA

"""

import os
import warnings

from ulamdyn._version import get_versions
from ulamdyn.data_loader import (
    GetCoords,
    GetCouplings,
    GetGradients,
    GetProperties,
)
from ulamdyn.data_writer import Geometries
from ulamdyn.descriptors import R2, RingParams, ZMatrix, SOAPDescriptor
from ulamdyn.kinetics import GetVelocities, KineticEnergy, VibrationalSpectra
from ulamdyn.nma.normal_mode_analysis import NormalModeAnalysis
from ulamdyn.unsup_models.geom_sampling import GeomSampling
from ulamdyn.unsup_models.geom_space import ClusterGeoms, DimensionReduction
from ulamdyn.unsup_models.traj_space import ClusterTrajs

__all__ = [
    "GetCoords",
    "GetGradients",
    "GetCouplings",
    "GetProperties",
    "Geometries",
    "R2",
    "ZMatrix",
    "RingParams",
    "SOAPDescriptor",
    "GetVelocities",
    "KineticEnergy",
    "VibrationalSpectra",
    "NormalModeAnalysis",
    "GeomSampling",
    "DimensionReduction",
    "ClusterGeoms",
    "ClusterTrajs",
]

warnings.filterwarnings("ignore", category=FutureWarning)

__title__ = "ULaMDyn"
__version__ = get_versions()["version"]
__author__ = "Max Pinheiro Jr"
__email__ = "maxjr82@gmail.com"
__maintainer__ = "Max Pinheiro Jr"
__license__ = "LGPLv3"
__copyright__ = "Copyright 2021 Max Pinheiro"

del get_versions

filedir = os.path.dirname(__file__)
