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

from ulamdyn.statistics import create_stats
from ulamdyn.wrappers.nma import NMAnalysis
from ulamdyn.wrappers.save_xyz import SaveXYZ
from ulamdyn.wrappers.bootstrap import Bootstrap
from ulamdyn.wrappers.save_datasets import SaveDataset
from ulamdyn.wrappers.sampling import SampleGeometries
from ulamdyn.wrappers.ring_analysis import RingAnalysis
from ulamdyn.wrappers.clustering import ClusteringAnalysis
from ulamdyn.wrappers.dim_reduction import DimensionReductionAnalysis

__all__ = ["main"]


class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        return argparse.HelpFormatter._split_lines(self, text, width)


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


def _get_parser():
    # Define command-line arguments. The commands can be read from a config.txt file.
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars="@", formatter_class=SmartFormatter
    )
    parser.add_argument(
        "--save_dataset",
        required=False,
        type=str,
        metavar="",
        choices=["all", "properties", "gradients", "nacs", "velocities", "vibspec"],
        default=None,
        help="R| Select data set to build from the MD outputs and save as csv file.\n Options: %(choices)s.",
    )

    parser.add_argument(
        "--save_xyz",
        required=False,
        type=str,
        metavar="",
        default=None,
        help='R| Write the requested data from all trajectories into XYZ file(s). The argument should be given as a comma separated list of strings,\n starting with geoms or grads and followed by optional subargs (example: "hops" or "hops,S21" or a query in the form "TRAJ==10").',
    )

    parser.add_argument(
        "--use_au",
        required=False,
        action="store_true",
        help="R| If selected, the XYZ Cartesian coordinates or gradients will be written in atomic units (useful for MLatom training).",
    )

    parser.add_argument(
        "--create_stats",
        required=False,
        type=str,
        metavar="",
        default=None,
        help="R| Generate a data set with basic statistics (mean, median, and std) for all the trajectories.\n Options: all, ekin, vibspec.",
    )

    parser.add_argument(
        "--bootstrap",
        required=False,
        type=str,
        metavar="",
        default=None,
        help="R| Compute the basic statistics and confidence intervals for the properties data set using the bootstrap approach.\n Options: n_repeats, and/or n_samples, and/or ci_level.",
    )

    pp = argparse.ArgumentParser(add_help=False)
    pp.add_argument(
        "--n_samples",
        required=False,
        type=int,
        metavar="",
        default=None,
        help="R| Number of samples randomly selected from the data set.",
    )

    pp.add_argument(
        "--time_step",
        required=False,
        type=float,
        metavar="",
        default=None,
        help="R| Size of the time step (dt) used to filter each trajectory in the data set.",
    )

    pp.add_argument(
        "--descriptor",
        required=False,
        type=str,
        metavar="",
        choices=["aXYZ", "R2", "inv-R2", "delta-R2", "RE", "Zmat", "delta-Zmat"],
        default="inv-R2",
        help="R| Descriptor used to represent molecular geometries.\n Options: %(choices)s. (default: %(default)s)",
    )

    pp.add_argument(
        "--use_mwc",
        required=False,
        action="store_true",
        help="R| Use mass weighted Cartesian coordinates to build R2-based descriptors.",
    )

    pp.add_argument(
        "--transform",
        required=False,
        type=str,
        metavar="",
        choices=["tanh", "sigmoid"],
        default=None,
        help="R| Apply a nonlinear transformation on delta type descriptors.\n Options: %(choices)s.",
    )

    pp.add_argument(
        "--data_scaler",
        required=False,
        type=str,
        metavar="",
        choices=["minmax", "standard", "robust", "norm"],
        default=None,
        help="R| Select the data rescaling method.\n Options: %(choices)s.",
    )

    pp.add_argument(
        "--n_cpus",
        required=False,
        type=int,
        metavar="",
        default=-1,
        help="R| Number of CPUs allocated for parallelization. (default: %(default)s)",
    )

    pp.add_argument(
        "--dist_metric",
        required=False,
        type=str,
        metavar="",
        choices=["euclidean", "seuclidean", "cosine", "correlation", "rmsd"],
        default="euclidean",
        help="R| Distance metric used for dimensionality reduction (Isomap and t-SNE) or clustering (Hierarchical).\n Options: %(choices)s. (default: %(default)s)",
    )

    pp.add_argument(
        "--kernel",
        required=False,
        type=str,
        metavar="",
        choices=["linear", "poly", "rbf", "laplacian", "sigmoid", "cosine"],
        default="rbf",
        help="R| Kernel function used for KPCA or Spectral clustering.\n Options: %(choices)s. (default: %(default)s)",
    )

    subparsers = parser.add_subparsers(title="Analysis", dest="command")

    ring_analysis = subparsers.add_parser(
        "ring_analysis",
        formatter_class=SmartFormatter,
        help="Cremer-Pople analysis for a cyclic substructure.",
    )

    ring_analysis.add_argument(
        "--atoms",
        required=True,
        type=str,
        metavar="",
        default=None,
        help="R| Atom indices in the connectivity order of the ring. The numbers must be passed as a comma separated list.",
    )

    ring_analysis.add_argument(
        "--stats_by",
        required=False,
        type=str,
        metavar="",
        default=None,
        help="R| Variables used for grouping data to compute the statistics.\n Options: time or state or time,state or class.",
    )

    dimred_analysis = subparsers.add_parser(
        "dim_reduction",
        parents=[pp],
        formatter_class=SmartFormatter,
        help="Dimensionality reduction analysis in molecular configuration space.",
    )
    dimred_analysis.add_argument(
        "--method",
        required=True,
        type=str,
        metavar="",
        default=None,
        help="R| Select algorithm for the analysis.\n Options: PCA, KPCA, Isomap, tSNE.",
    )
    dimred_analysis.add_argument(
        "--n_dim",
        required=False,
        type=int,
        metavar="",
        default=2,
        help="R| Number of dimensions of the reduced data set. (default: %(default)s)",
    )
    dimred_analysis.add_argument(
        "--perplexity",
        required=False,
        type=float,
        metavar="",
        default=40,
        help="R| Perplexity parameters used in the t-SNE algorithm. (default: %(default)s)",
    )

    clustering_analysis = subparsers.add_parser(
        "clustering",
        parents=[pp],
        formatter_class=SmartFormatter,
        help="Perform cluster analysis in geometry or trajectory space.",
    )
    clustering_analysis.add_argument(
        "--space",
        required=True,
        type=str,
        metavar="",
        choices=["geoms", "trajs"],
        default=None,
        help="R| Select the type of data in which the clustering analysis will be performed.\n Options: %(choices)s.",
    )
    clustering_analysis.add_argument(
        "--method",
        required=True,
        type=str,
        metavar="",
        default=None,
        help="R| Select model to perform clustering analysis.\n Options: K-means (for geoms or trajs), GMM, Hierarchical, Spectral.",
    )
    clustering_analysis.add_argument(
        "--n_clusters",
        required=False,
        type=str,
        metavar="",
        default="3",
        help="R| Number of clusters in which the data set will be grouped. If value is equal to best,\n the optimal number of clusters will be searched. (default: %(default)s)",
    )
    geom_sampling = subparsers.add_parser(
        "sampling",
        parents=[pp],
        formatter_class=SmartFormatter,
        help="Sample new molecular geometries using Gaussian Mixture model.",
    )
    geom_sampling.add_argument(
        "--n_clusters",
        required=False,
        type=str,
        metavar="",
        default="3",
        help="R| Number of mixed gaussians from which new geometries will be sampled. If value is equal to best,\n the optimal number of clusters will be searched. (default: %(default)s)",
    )
    geom_sampling.add_argument(
        "--n_new_geoms",
        required=False,
        type=int,
        metavar="",
        default=None,
        help="R| Number of geometries to be generated by the Gaussian Mixture algorithm.",
    )
    geom_sampling.add_argument(
        "--sample_from",
        required=False,
        type=str,
        metavar="",
        default="0",
        help='R| Select the cluster from which new geometries will be sampled. The argument can be the cluster ID (an integer)\n or a query string based on the properties dataset (e.g., "DE21<=0.5").',
    )
    nm_analysis = subparsers.add_parser(
        "nma",
        formatter_class=SmartFormatter,
        help="Perform Normal Mode analysis to describe the molecular motion.",
    )
    nm_analysis.add_argument(
        "--vib_file",
        required=False,
        type=str,
        metavar="",
        default="freq.molden",
        help="R| Name of the MOLDEN input file that contains the normal modes for the reference geometry. (default: %(default)s)",
    )
    nm_analysis.add_argument(
        "--time_intervals",
        required=False,
        type=str,
        metavar="",
        default="",
        help="R| List of time intervals on which the average will be calculated (example: 0-250,251-500). (default: %(default)s)",
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

    return args


def main():
    args = _get_parser()
    _check_geom_file()

    start = time.time()

    keywords_dict = vars(args)

    if args.save_dataset is not None:
        data_to_save = keywords_dict.get("save_dataset")
        SaveDataset.run(data_to_save)

    if args.save_xyz is not None:
        SaveXYZ.run(**keywords_dict)

    if args.create_stats is not None:
        _ = create_stats(keywords_dict.get("create_stats"), save_csv=True)

    if args.bootstrap is not None:
        Bootstrap.run(**keywords_dict)

    if args.command is not None:
        if args.command == "sampling":
            print("=" * 55)
            print("Molecular geometries will be sampled using Gaussian Mixture")
            print("=" * 55)
            print("")
            SampleGeometries.run(**keywords_dict)
        else:
            analysis_type = args.command.split("_")[0]
            if analysis_type == "dim":
                analysis_type = "dimension reduction"
            print("=" * 55)
            print("The {} analysis will be performed".format(analysis_type))
            print("=" * 55)
            print("")

            analysis_type = analysis_type.title().replace(" ", "")
            if analysis_type == "Nma":
                analysis_type = "NM"
            func = eval(analysis_type + "Analysis.run")
            func(**keywords_dict)

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
