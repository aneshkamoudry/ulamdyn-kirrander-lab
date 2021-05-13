#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup file for the ulamdyn package.
"""

from __future__ import with_statement
from __future__ import absolute_import

import os
from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Unsupervised learning for molecular dynamics data'
LONG_DESCRIPTION = 'A package that provide a set of methods for the preprocessing, statistical, and unsupervised learning analysis of data from molecular dynamics simulations.'

# Setting up
setup(
    name="ulamdyn",
    version=VERSION,
    author="Max Pinheiro Jr",
    author_email="<maxjr82@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=['numpy', 'rmsd', 'pandas', 'scikit-learn'],
    keywords=['python', 'chemistry', 'dimensionality reduction', 'clustering', 'molecular dynamics'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Computational chemistry",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
    python_requires=">=3.7",
)

