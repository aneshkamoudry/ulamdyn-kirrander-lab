# -*- coding: utf-8 -*-
"""
  ULaMDyn - Unsupervised Learning analysis for Molecular Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ULaMDyn is a python package built on top of sklearn designed to perform 
data preprocessing, statistical and unsupervised learning analysis of 
(non-adiabatic) molecular dynamics simulations.

ULaMDyn consists of five general modules:

- Data_Loader
- Data_Writer
- Statistics
- Kinetics
- Descriptors
- Unsup_Models

"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import os
import pkg_resources

def export(func):
    if callable(func) and hasattr(func, '__name__'):
        globals()[func.__name__] = func
    try:
        __all__.append(func.__name__)
    except NameError:
        __all__ = [func.__name__]
    return func

from ulamdyn.data_loader import *
from ulamdyn.data_writer import *
from ulamdyn.descriptors import *
from ulamdyn.statistics import *
from ulamdyn.kinetics import *
from ulamdyn.unsup_models import *
from ulamdyn.utilities import *

filedir = os.path.dirname(__file__)

__title__ = 'ULaMDyn'
__version__ = '0.0.2'
__author__ = 'Max Pinheiro Jr'
__email__ = 'maxjr82@gmail.com'
__maintainer__ = 'Max Pinheiro Jr'
__license__ = 'LGPLv3'
__copyright__ = 'Copyright 2021 Max Pinheiro'
