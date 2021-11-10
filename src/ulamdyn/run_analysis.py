# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: March 10, 2021
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
    with_statement,
)

import os
import sys
import time
import argparse
import numpy as np

from ulamdyn.data_loader import *
from ulamdyn.data_writer import *
from ulamdyn.descriptors import *
from ulamdyn.statistics import *
from ulamdyn.unsup_models import *
from ulamdyn.interface import *

__all__ = ["main"]


def _check_geom_file():
    if not os.path.isfile("geom.xyz"):
        print("\n--------------------------------------------------------")
        print("ERROR:                                             \n")
        print("The geom.xyz file was not found.")
        print("Please provide a reference geometry (geom.xyz) in the ")
        print("working directory.")
        print("Aborting execution.")
        print("--------------------------------------------------------")
        sys.exit()


def main():

    # Define command-line arguments. The commands can be read from a config.txt file.
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument(
        "--save_dataset",
        required=False,
        type=str,
        metavar="all | gradients | nacs | vibspec",
        default=None,
        help="Type of data set (properties + descriptors) to build from MD outputs\
                              and save as csv file.",
    )
    parser.add_argument(
        "--save_xyz_hops",
        required=False,
        type=str,
        metavar="",
        default=None,
        help="Write a single XYZ file with all hopping geometries for a \
                              given pair of states.",
    )
    parser.add_argument(
        "--create_stats",
        required=False,
        type=str,
        metavar="all | ekin | vibspec",
        default=None,
        help="Generate a data set with basic statistics (mean, median \
                              and std) for all the trajectories.",
    )
    parser.add_argument(
        "--bootstrap",
        required=False,
        type=str,
        metavar="",
        default=None,
        help="Run the bootstrap algorithm for the properties data set \
                              and save a new data with basic statistics (mean, median \
                              and std) and confidence intervals.",
    )
    parser.add_argument(
        "--descriptor",
        required=False,
        type=str,
        metavar="",
        default="inv-R2",
        help="Select the molecular descriptor to be used in the unsupervised\
                              learning analysis.",
    )

    parser.add_argument(
        "--transform",
        required=False,
        type=str,
        metavar="",
        default=None,
        help="Apply a nonlinear transformation (sigmoid or tanh) on delta type of\
              descriptors learning analysis.",
    )

    parser.add_argument(
        "--data_scaler",
        required=False,
        type=str,
        metavar="",
        default=None,
        help="Method to rescale the data set before applying the \
                              unsupervised learning model.",
    )
    parser.add_argument(
        "--n_samples",
        required=False,
        type=int,
        metavar="",
        default=None,
        help="Number of samples randomly selected from the data set.",
    )
    parser.add_argument(
        "--dim_reduction",
        required=False,
        type=str,
        metavar="",
        default=None,
        help="Select a model for the dimensionality reduction analysis:\
                              PCA, KPCA, Isomap or tSNE.",
    )
    parser.add_argument(
        "--n_dim",
        required=False,
        type=int,
        metavar="",
        default=2,
        help="Number of dimensions of the reduced data set.",
    )
    parser.add_argument(
        "--kernel",
        required=False,
        type=str,
        metavar="",
        default="rbf",
        help="Kernel function used for KPCA method.",
    )
    parser.add_argument(
        "--perplexity",
        required=False,
        type=float,
        metavar="",
        default=40,
        help="Perplexity parameters used in the t-SNE algorithm.",
    )
    parser.add_argument(
        "--clustering",
        required=False,
        type=str,
        metavar="",
        default=None,
        help="Model used for clustering analysis: K-means, Hierarchical or Spectral.",
    )
    parser.add_argument(
        "--n_clusters",
        required=False,
        type=str,
        metavar="",
        default="3",
        help="Number of clusters in which the data set will be grouped.",
    )
    parser.add_argument(
        "--n_cpus",
        required=False,
        type=int,
        metavar="",
        default=-1,
        help="Number of CPUs allocated for the parallelization of \
                              the unsupervised learning tasks. If -1 all available \
                              processors will be used.",
    )

    # If no command-line arguments are present, config file is parsed
    config_file = "config.txt"
    if len(sys.argv) == 1:
        if os.path.isfile(config_file):
            args = parser.parse_args(["@" + config_file])
        else:
            args = parser.parse_args(["--help"])
    else:
        args = parser.parse_args()

    _check_geom_file()

    start = time.time()

    if args.save_dataset is not None:
        save_data(args.save_dataset)

    if args.save_xyz_hops is not None:
        save_xyz_hoppings(args.save_xyz_hops)

    if args.create_stats is not None:
        data_dict = create_stats(args.create_stats, save_csv=True)

    if args.bootstrap is not None:
        run_bootstrap(args)

    if args.dim_reduction is not None:
        run_dim_reduction(args)

    if args.clustering is not None:
        run_clustering(args)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\n---------------------------------------------------")
    print(
        "Total execution time - {:0>2}:{:0>2}:{:05.2f}".format(
            int(hours), int(minutes), seconds
        )
    )
    print("---------------------------------------------------\n")

    print("Good Bye!")
