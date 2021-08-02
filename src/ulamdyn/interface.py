## Author: Max Pinheiro Jr <maxjr82@gmail.com>
## Date: May 13, 2021
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

try:
    import modin.pandas as pd
    import ray

    ray.shutdown()
    ray.init()
except:
    import pandas as pd

from ulamdyn.data_loader import *
from ulamdyn.data_writer import *
from ulamdyn.kinetics import *
from ulamdyn.descriptors import *
from ulamdyn.statistics import *
from ulamdyn.unsup_models import *


def get_kinetic_energies(n_atoms=None):

    ke = KineticEnergy(n_atoms=n_atoms)
    df_ekin = ke.build_dataframe()
    return df_ekin


def get_properties_data(rmsd_vec=None):

    try:
        df = pd.read_csv("all_properties.csv")
        print("Loading properties data from existing csv...\n")
    except:
        gp = GetProperties()
        df = gp.energies()
        df = gp.oscillator_strength()
        df = gp.populations()

    df_ekin = get_kinetic_energies()
    df = pd.concat([df, df_ekin], axis=1)

    if rmsd_vec is not None:
        try:
            df["RMSD"] = rmsd_vec
        except:
            s1, s2 = len(rmsd_vec), df.shape[0]
            print("--------------------------------------------------------")
            print("ERROR: \n")
            print("The size of the coordinates ({}) and properties ({})".format(s1, s2))
            print("data sets does not match.")
            print("The RMSD can not be added to the properties data set.")
            print("--------------------------------------------------------")

    return df


def save_data(data_to_save):
    if data_to_save == "all":
        print("Saving the full XYZ coordinates dataframe...\n")
        gc = GetCoords()
        gc.read_all_trajs()
        gc.align_geoms
        gc.save_csv
        print("Saving the full properties dataframe...\n")
        gp = GetProperties()
        df = gp.energies()
        df = gp.oscillator_strength()
        df = gp.populations()
        gp.save_csv
        print("Saving the R2 descriptor dataframe...\n")
        r2 = R2()
        all_geoms = gc.xyz.copy()
        df = r2.build_descriptor(all_geoms, save_csv=True)
        print("Saving the Z-Matrix descriptor dataframe...\n")
        zmt = ZMatrix()
        df = zmt.build_descriptor(all_geoms, save_csv=True)

    elif data_to_save == "gradients":
        print("Saving the XYZ gradients for each available state as dataframes...\n")
        gg = GetGradients()
        gg.build_dataframe(save_csv=True)


def build_descriptor(descriptor, getcoords_obj):
    all_aligned_geoms = getcoords_obj.xyz
    # getcoords_obj.xyz is a variable of the class object
    # that stores all XYZ coordinates as a numpy array of
    # dimension [n_geoms, n_atoms, 3]
    if descriptor == "aXYZ":
        getcoords_obj.build_dataframe()
        df_xyz = getcoords_obj.dataset
        df_xyz.to_csv(descriptor + ".csv", index=False)
        return df_xyz
    elif descriptor in ["R2", "inv-R2", "delta-R2", "RE"]:
        r2 = R2()
        df_r2 = r2.build_descriptor(all_aligned_geoms, descriptor)
        df_r2.to_csv(descriptor + ".csv", index=False)
        return df_r2
    elif descriptor in ["Zmat", "delta-Zmat"]:
        zmt = ZMatrix()
        dfs_dict = {
            "Zmat": zmt.build_descriptor(all_aligned_geoms),
            "delta-Zmat": zmt.build_descriptor(all_aligned_geoms, delta=True),
        }
        df_zmt = dfs_dict[descriptor]
        df_zmt.to_csv(descriptor + ".csv", index=False)
        return df_zmt
    else:
        print("-----------------------------------------------------")
        print("ERROR: \n")
        print("Descriptor not recognized or implemented!\n")
        print("Please select one of the available descriptors:")
        print("aXYZ, R2, inv-R2, delta-R2, RE, Zmat or delta-Zmat.")
        print("-----------------------------------------------------")
        sys.exit()


def run_dim_reduction(args):
    # Step 1: Load XYZ data from all trajectories and align coordinates
    gc = GetCoords()
    gc.read_all_trajs()
    gc.align_geoms
    rmsd_vals = gc.rmsd

    # Step 2: create the dataset to apply the dimensionality reduction model.
    df = build_descriptor(args.descriptor, gc)

    # Step 3: build the dataset of properties that can be used for colormap.
    df_props = get_properties_data(rmsd_vals)

    # Step 4: instanciate the dimensionality reduction class
    dimred = DimensionalityReduction(
        data=df, n_samples=args.n_samples, scaler=args.data_scaler, n_cpus=args.n_cpus
    )
    model = args.dim_reduction.lower().strip()

    # Step 5: check for the available models and run the calculation
    if model == "pca":
        df_reduced = dimred.pca(n_components=args.n_dim, calc_error=True)
    elif model == "kpca":
        df_reduced = dimred.kpca(n_components=args.n_dim, kernel=args.kernel)
    elif model == "isomap":
        df_reduced = dimred.isomap(n_components=args.n_dim, calc_error=True)
    elif model == "tsne":
        df_reduced = dimred.tsne(n_components=args.n_dim, perplexity=args.perplexity)
    else:
        print("--------------------------------------------------------")
        print("ERROR:                                             \n")
        print("Model type not recognized or not implemented!")
        print("Please select one of the available methods:")
        print("PCA, KPCA, Isomap or t-SNE.")
        print("--------------------------------------------------------")
        sys.exit()

    # Step 5: Save a csv file with the merged datasets (reduced + properties)
    print("Saving the merged (reduced + properties) data sets...\n")
    df_reduced = df_reduced.merge(
        df_props, left_index=True, right_index=True, how="left"
    )
    csv_name = model + "_ndim" + str(args.n_dim) + "_"
    csv_name += args.descriptor.lower() + ".csv"
    df_reduced.to_csv(csv_name, header=True, index=True, index_label="index")


def run_clustering(args):
    # Step 1: Load XYZ data from all trajectories and align coordinates
    gc = GetCoords()
    gc.read_all_trajs()
    gc.align_geoms
    rmsd_vals = gc.rmsd

    # Step 2: create the dataset to apply the dimensionality reduction model.
    df = build_descriptor(args.descriptor, gc)

    # Step 3: build the dataset of properties that can be used for colormap.
    df_props = get_properties_data(rmsd_vals)

    # Step 4: instanciate the dimensionality reduction class
    cluster = Clustering(
        data=df, n_samples=args.n_samples, scaler=args.data_scaler, n_cpus=args.n_cpus
    )
    model = args.clustering.lower().strip()
    model = model.replace("-", "")
    n_clusters = args.n_clusters

    if n_clusters.lower() != "best":
        n_clusters = list(map(int, n_clusters.split(",")))
        if len(n_clusters) == 1:
            n_clusters = n_clusters[0]

    if model == "kmeans":
        df_labels = cluster.kmeans(n_clusters=n_clusters)
        print("Saving data set with all cluster labels...\n")
        df_labels.to_csv(
            "kmeans_labels.csv", header=True, index=True, index_label="index"
        )
    elif model == "spectral":
        df_labels = cluster.spectral(n_clusters=n_clusters)
        print("Saving data set with all cluster labels...\n")
        df_labels.to_csv(
            "spectral_labels.csv", header=True, index=True, index_label="index"
        )
    else:
        print("--------------------------------------------------------")
        print("ERROR:                                             \n")
        print("Model type not recognized or not implemented!")
        print("Please select one of the available methods:")
        print("K-Means or Spectral.")
        print("--------------------------------------------------------")
        sys.exit()

    # These lines are used to recover the numerical info of the number of clusters
    # for the cases in which the original n_clusters variable is a list or 'best'
    col_labels = [s for s in df_labels.columns.tolist() if "labels" in s][0]
    n_clusters = len(df_labels[col_labels].unique())

    print("Saving properties data sets with cluster labels included...\n")
    df_props = df_props.merge(df_labels, left_index=True, right_index=True, how="right")
    csv_basename = model + "_k" + str(n_clusters)
    csv_name = csv_basename + "_properties.csv"
    df_props.to_csv(csv_name, header=True, index=True, index_label="index")

    print("Creating statistics for properties data based on clusters...\n")
    groupby_clusters = df_labels.columns.tolist()
    df_prop_stats = aggregate_data(df_props, groupby_clusters)
    csv_name = csv_basename + "_stats_properties.csv"
    df_prop_stats.to_csv(csv_name, header=True, index=True, index_label="index")

    print("Creating statistics for Z-Matrix data based on clusters...\n")
    try:
        df_zmt = pd.read_csv("all_geoms_zmatrix.csv")
    except:
        df_zmt = build_descriptor("Zmat", gc)

    df_zmt = df_zmt.merge(df_labels, left_index=True, right_index=True, how="right")
    df_zmt_stats = aggregate_data(df_zmt, groupby_clusters)
    csv_name = csv_basename + "_stats_zmatrix.csv"
    df_zmt_stats.to_csv(csv_name, header=True, index=True, index_label="index")

    print("Saving average XYZ geometry for each clusters...")
    gc.build_dataframe()
    df_xyz = gc.dataset
    df_xyz = df_xyz.merge(df_labels, left_index=True, right_index=True, how="right")
    df_xyz_mean = df_xyz.groupby(groupby_clusters).mean()
    xyz_mean = df_xyz_mean.values.reshape(n_clusters, -1, 3)
    atom_labels = gc.labels.reshape(-1, 1)
    geoms = Geometries(atom_labels)
    out_name = "Geoms_centroids_" + model + ".xyz"
    geoms.save_xyz(xyz_mean, out_name=out_name)

    print("                             ")
    print("Clustering analysis finished!")
    print("                             ")


def save_xyz_hoppings(states_pair):
    # 1) Load the XYZ coordinates from all trajectories
    gc = GetCoords()
    gc.read_all_trajs()
    gc.align_geoms

    # 2) Generate the dataframe with all properties
    df_props = get_properties_data(gc.rmsd)

    col_hopping = "Hops_" + states_pair
    if col_hopping in df_props.columns.tolist():
        indices = df_props[df_props[col_hopping] == 1].index.tolist()
        df_props = df_props[df_props[col_hopping] == 1].reset_index(drop=True)
        df_props = df_props.round(4)
        hopping_geoms = gc.xyz[indices].copy()
        atom_labels = gc.labels.reshape(-1, 1)
        add_property = [states_pair.replace("S", "DE")]
        if "RMSD" in df_props.columns:
            add_property.append("RMSD")
        out_name = "Geoms_Hopping_" + states_pair + ".xyz"
        geoms = Geometries(atom_labels, add_property)
        geoms.save_xyz(hopping_geoms, df_props, out_name)
    else:
        print("-----------------------------------------------------")
        print("There is no hopping for the selected pair of states.")
        print("-----------------------------------------------------")


def run_bootstrap(args):

    print("Loading the all properties data...\n")
    try:
        df = pd.read_csv("all_properties.csv")
    except:
        gp = GetProperties()
        df = gp.energies()
        df = gp.oscillator_strength()
        df = gp.populations()

    print("Running the bootstrap algorithm...\n")

    bootstrap_options = args.bootstrap.split(",")

    save_csv = True if "save" in bootstrap_options else False

    # Setting some default values for the bootstrap statistics
    n_samples = len(df["TRAJ"].unique())
    n_repeats = 1000
    ci_level = 95

    if len(bootstrap_options) >= 3:
        n_samples, n_repeats = list(map(int, bootstrap_options[:2]))
        # Check if the CI level is provided as input.
        # It must be the third option in the list.
        try:
            ci_level = int(bootstrap_options[2])
        except ValueError:
            pass

    elif len(bootstrap_options) == 2:
        if save_csv:
            n_repeats = int(bootstrap_options[0])
        else:
            n_repeats, ci_level = list(map(int, bootstrap_options[:1]))

    elif len(bootstrap_options) == 1:
        if not save_csv:
            n_repeats = int(bootstrap_options)

    print("The following parameters will be used in the bootstrap:\n")
    print("   number of trajectories = {:<20}".format(n_samples))
    print("        number of repeats = {:<20}".format(n_repeats))
    print("      confidence interval = {:<20}%".format(ci_level))
    print(" ")

    df_bootstrap = bootstrap(
        df, n_samples=n_samples, n_repeats=n_repeats, save_csv=save_csv
    )

    print("Creating statistics for the bootstrapped data...\n")
    df_stats = create_bootstrap_stats(df_bootstrap, ci_level)
    df_stats.to_csv("stats_bootstrap.csv", index=False)
