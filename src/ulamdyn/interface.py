# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: May 13, 2021
from __future__ import (
    absolute_import,
    print_function,
    unicode_literals,
    with_statement,
)

import os
import sys
import numpy as np

try:
    import modin.pandas as pd

except:
    import pandas as pd

from ulamdyn.data_loader import *
from ulamdyn.data_writer import *
from ulamdyn.kinetics import *
from ulamdyn.descriptors import *
from ulamdyn.statistics import *
from ulamdyn.unsup_models.geom_space import *
from ulamdyn.unsup_models.traj_space import *
from ulamdyn.unsup_models.dist_metrics import calc_rmsd
from ulamdyn.nx_utils import *


def get_kinetic_energies(n_atoms=None):

    ke = KineticEnergy(n_atoms=n_atoms)
    df_ekin = ke.build_dataframe()
    return df_ekin


def get_properties_data(rmsd_vec=None):

    try:
        df = pd.read_csv("all_properties.csv")
        print("\nLoading properties data from existing csv...\n")
    except FileNotFoundError:
        gp = GetProperties()
        all_properties = [
            "energies",
            "oscillator_strength",
            "mcscf_coefs",
            "populations",
            "nac_norm",
        ]
        for prop in all_properties:
            _ = eval("gp." + prop + "()")
        df = gp.dataset

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


def save_data(args):
    data_to_save = args.save_dataset
    if data_to_save == "all":
        print("Saving the full XYZ coordinates dataframe...\n")
        gc = GetCoords()
        gc.read_all_trajs()
        gc.align_geoms
        gc.save_csv
        print("Saving the R2 descriptor dataframe...\n")
        r2 = R2()
        all_geoms = gc.xyz.copy()
        df = r2.build_descriptor(all_geoms, save_csv=True)
        print("Saving the Z-Matrix descriptor dataframe...\n")
        zmt = ZMatrix()
        df = zmt.build_descriptor(all_geoms, save_csv=True)
        print("Saving the full properties dataframe...\n")
        df = get_properties_data()
        df.to_csv("all_properties.csv", index=False)

    elif data_to_save.lower() == "gradients":
        print("Saving the XYZ gradients for each available state as dataframes...\n")
        gg = GetGradients()
        gg.build_dataframe(save_csv=True)

    elif data_to_save.lower() == "nacs":
        print("Saving the NACs for each state pair as separated dataframes...\n")
        gnac = GetCouplings()
        gnac.build_dataframe(save_csv=True)

    elif data_to_save.lower() == "vibspec":
        print("Saving the vibrational (power) spectra for all MD trajectories...\n")
        vs = VibrationalSpectra()
        df_spec = vs.build_dataframe()
        gp = GetProperties()
        df_prop = gp.energies()
        if df_spec.shape[0] == df_prop.shape[0]:
            df_spec.insert(1, "State", df_prop["State"].values)
        df_spec.to_csv("all_vibrational_spectra.csv", index=False, header=True)


def build_descriptor(args, getcoords_obj):
    descriptor = args.descriptor
    transform = args.transform
    mwc = args.use_mwc

    all_aligned_geoms = getcoords_obj.xyz
    # getcoords_obj.xyz is a variable of the class object
    # that stores all XYZ coordinates as a numpy array of
    # dimension [n_geoms, n_atoms, 3]
    if descriptor == "aXYZ":
        getcoords_obj.build_dataframe()
        df_xyz = getcoords_obj.dataset
        select_cols = df_xyz.filter(regex=r"^[x,y,z]", axis=1).columns.tolist()
        # Features should contain only the Cartesian coordinates
        df_xyz = df_xyz[select_cols]
        df_xyz.to_csv(descriptor + ".csv", index=False)
        return df_xyz
    elif descriptor in ["R2", "inv-R2", "delta-R2", "RE"]:
        r2 = R2(mwc)
        df_r2 = r2.build_descriptor(all_aligned_geoms, descriptor)
        df_r2.to_csv(descriptor + ".csv", index=False)
        return df_r2
    elif descriptor in ["Zmat", "delta-Zmat"]:
        zmt = ZMatrix()
        dfs_dict = {
            "Zmat": zmt.build_descriptor(all_aligned_geoms),
            "delta-Zmat": zmt.build_descriptor(
                all_aligned_geoms, delta=True, apply_to_delta=transform
            ),
        }
        df_zmt = dfs_dict[descriptor]
        df_zmt.to_csv(descriptor + ".csv", index=False)
        return df_zmt
    else:
        print("-----------------------------------------------------")
        print("Input error: \n")
        print("Descriptor not recognized or implemented!\n")
        print("Please select one of the available descriptors:")
        print("aXYZ, R2, inv-R2, delta-R2, RE, Zmat or delta-Zmat.")
        print("-----------------------------------------------------")
        sys.exit()


def run_ring_analysis(args):
    atom_list_str = args.atoms
    if "," not in atom_list_str:
        print("-----------------------------------------------------")
        print("Input error: \n")
        print("The indices of the atoms composing the ring must be")
        print("provided as a comma separated list.")
        print("-----------------------------------------------------")
        sys.exit()
    else:
        atom_indices = list(map(int, atom_list_str.split(",")))
        ra = RingParams(ring_atom_ind=atom_indices)
        df = ra.build_dataframe(save_csv=True)
        if args.stats_by is not None:
            vars_to_group = args.stats_by.lower().split(",")
            if "state" in vars_to_group:
                gp = GetProperties()
                df_en = gp.energies()
                try:
                    df.insert(2, "state", df_en["State"].values)
                except ValueError as ve:
                    print(
                        "-------------------------------------------------------------"
                    )
                    print("Mismatch error: \n")
                    print(
                        "The state labels from the energy data set cannot be inserted"
                    )
                    print("in the ring_params data set due to incompatible sizes.\n")
                    print("Size energy data = {}".format(df_en.shape[0]))
                    print("Size ring_params data = {}".format(df.shape[0]))
                    print(
                        "-------------------------------------------------------------"
                    )
                    sys.exit()
            df_stats = aggregate_data(df, vars_to_group)
            df_stats.to_csv("stats_ring_params.csv", index=False, header=True)


def run_dim_reduction(args):
    # Step 1: Load XYZ data from all trajectories and align coordinates
    gc = GetCoords()
    gc.read_all_trajs()
    gc.align_geoms
    rmsd_vals = gc.rmsd

    # Step 2: create the input features dataset from molecular geometries.
    df = build_descriptor(args, gc)

    # Step 3: build the dataset of properties that can be used for colormap.
    df_props = get_properties_data(rmsd_vals)

    # Step 4: instanciate the dimensionality reduction class
    dimred = DimensionReduction(
        data=df, n_samples=args.n_samples, scaler=args.data_scaler, n_cpus=args.n_cpus
    )
    model = args.method.lower().strip()
    metric = args.dist_metric  # .lower().strip()

    if metric == "rmsd":
        if args.descriptor != "aXYZ":
            print("*********************************************************")
            print("WARNING:                                             \n")
            print("RMSD can be used only with the aXYZ descriptor.")
            print("The Euclidean distance will be used instead as default.")
            print("*********************************************************")
            metric = "euclidean"
        else:
            metric = calc_rmsd

    # Step 5: check for the available models and run the calculation
    if model == "pca":
        df_reduced = dimred.pca(n_components=args.n_dim, calc_error=True)
    elif model == "kpca":
        df_reduced = dimred.kpca(n_components=args.n_dim, kernel=args.kernel)
    elif model == "isomap":
        df_reduced = dimred.isomap(
            n_components=args.n_dim, metric=metric, calc_error=True
        )
    elif model == "tsne":
        df_reduced = dimred.tsne(
            n_components=args.n_dim, metric=metric, perplexity=args.perplexity
        )
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
    # Step 1: Load XYZ data from all trajectories and align geometries
    gc = GetCoords()
    gc.read_all_trajs()
    gc.align_geoms
    rmsd_vals = gc.rmsd

    # Step 2: create the dataset to apply the dimensionality reduction model.
    df = build_descriptor(args, gc)

    # Step 3: build the dataset of properties that can be used for color map.
    df_props = get_properties_data(rmsd_vals)

    # Step 4: create instance for the clustering method.
    cluster = ClusterGeoms(
        data=df, n_samples=args.n_samples, scaler=args.data_scaler, n_cpus=args.n_cpus
    )
    model = args.method.lower().strip()
    model = model.replace("-", "")
    n_clusters = args.n_clusters

    if n_clusters.lower() != "best":
        n_clusters = list(map(int, n_clusters.split(",")))
        if len(n_clusters) == 1:
            n_clusters = n_clusters[0]

    if model == "kmeans":
        df_labels = cluster.kmeans(n_clusters=n_clusters)
    elif model == "hierarchical":
        df_labels = cluster.hierarchical(n_clusters=n_clusters)
    elif model == "spectral":
        df_labels = cluster.spectral(n_clusters=n_clusters, affinity=args.kernel)
    else:
        print("--------------------------------------------------------")
        print("ERROR:                                             \n")
        print("Model {} not recognized or not implemented.".format(model))
        print("Please select one of the available methods:")
        print("K-Means, Hierarchical or Spectral.")
        print("--------------------------------------------------------")
        sys.exit()

    print("Saving data set with all cluster labels...\n")
    df_labels.to_csv(
        model + "_labels.csv", header=True, index=True, index_label="index"
    )

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
    except FileNotFoundError:
        zmt = ZMatrix()
        df_zmt = zmt.build_descriptor(gc.xyz)

    df_zmt = df_zmt.merge(df_labels, left_index=True, right_index=True, how="right")
    df_zmt_stats = aggregate_data(df_zmt, groupby_clusters)
    csv_name = csv_basename + "_stats_zmatrix.csv"
    df_zmt_stats.to_csv(csv_name, header=True, index=True, index_label="index")

    print("Saving geometries for each clusters...\n")
    gc.build_dataframe()
    atom_labels = gc.labels.reshape(-1, 1)

    df_xyz = gc.dataset.copy()
    df_xyz = df_xyz.merge(df_labels, left_index=True, right_index=True, how="right")

    info2xyz = ["State"]
    if "RMSD" in df_props.columns:
        info2xyz.append("RMSD")

    for cluster in range(n_clusters):
        select_cluster = col_labels + "==" + str(cluster)
        df_temp = df_xyz.query(select_cluster).sort_values(by=["time"])
        df_temp = df_temp.drop(["TRAJ", "time", col_labels], axis=1)
        n_geoms = df_temp.shape[0]
        xyz_cluster = df_temp.values.reshape(n_geoms, -1, 3)
        df_props_cluster = df_props.query(select_cluster).sort_values(by=["time"])
        df_props_cluster = df_props_cluster.reset_index(drop=True)
        geoms = Geometries(atom_labels, info2xyz)
        out_name = model + "_geoms_cluster" + str(cluster + 1) + ".xyz"
        geoms.save_xyz(xyz_cluster, df_props_cluster, out_name=out_name)

    print("Saving average geometries corresponding to the clusters' centroids...\n")
    df_xyz = df_xyz.drop(["TRAJ", "time"], axis=1)
    df_xyz_mean = df_xyz.groupby(groupby_clusters).mean()
    xyz_mean = df_xyz_mean.values.reshape(n_clusters, -1, 3)

    geoms = Geometries(atom_labels)
    out_name = model + "_geoms_centroids.xyz"
    geoms.save_xyz(xyz_mean, out_name=out_name)

    print("*" * 30)
    print("Clustering analysis finished!")
    print("*" * 30)


def save_xyz(args):

    save_options = args.save_xyz.split(",")
    atomic_units = args.use_au
    # 1) Load the XYZ coordinates from all trajectories
    gc = GetCoords()
    gc.read_all_trajs()
    gc.align_geoms
    atom_labels = gc.labels.reshape(-1, 1)
    n_atoms = len(atom_labels)

    # 2) Generate the dataframe with all properties
    df_props = get_properties_data(gc.rmsd)
    df_props = df_props.round(4)
    add_property = []
    if "RMSD" in df_props.columns:
        add_property.append("RMSD")

    if save_options[0].lower() == "hops":
        col_hoppings = [col for col in df_props.columns if "Hops" in col]
        for col in col_hoppings:
            indices = df_props[df_props[col] == 1].index.tolist()
            if len(indices) > 0:
                df_props_hops = (
                    df_props[df_props[col] == 1].copy().reset_index(drop=True)
                )
                hopping_geoms = gc.xyz[indices].copy()
                states_pair = col.split("_")[1]
                egap_info = [states_pair.replace("S", "DE")]
                out_name = "Geoms_Hopping_" + states_pair + ".xyz"
                geoms = Geometries(atom_labels, add_property + egap_info)
                geoms.save_xyz(hopping_geoms, df_props, out_name)
    elif save_options[0].lower() == "geoms":
        all_geoms = gc.xyz.copy()
        if atomic_units:
            all_geoms *= 1 / BOHR_TO_ANG
        geoms = Geometries(atom_labels, add_property)
        geoms.save_xyz(all_geoms, df_props, "all_geometries.xyz")
    elif save_options[0].lower() == "grads":
        empty_labels = np.array([[""] for i in range(n_atoms)])
        gg = GetGradients()
        gg.build_dataframe()
        for state in gg.all_grads.keys():
            grads = gg.all_grads[state].copy()
            if atomic_units:
                grads *= BOHR_TO_ANG / HARTREE_TO_eV
            out_name = "all_gradients_" + state.lower() + ".xyz"
            geoms = Geometries(empty_labels)
            geoms.save_xyz(grads, df_props, out_name)
    else:
        print("-----------------------------------------------------")
        print("ERROR: \n")
        print("Save option not recognized or implemented!\n")
        print("Please select one of the following options:")
        print("hops, geoms or grads")
        print("-----------------------------------------------------")
        sys.exit()


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
    df = get_properties_data()

    print("Running the bootstrap algorithm...\n")

    bootstrap_options = args.bootstrap.split(",")

    save_csv = True if "save" in bootstrap_options else False

    # Setting some default values for the bootstrap statistics
    n_samples = len(df["TRAJ"].unique())
    n_repeats = 1000
    ci_level = 95

    if len(bootstrap_options) >= 3:
        n_repeats, n_samples = list(map(int, bootstrap_options[:2]))
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
