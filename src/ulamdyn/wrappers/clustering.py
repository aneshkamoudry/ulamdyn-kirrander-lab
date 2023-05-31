# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: June 3, 2022

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
from ulamdyn.unsup_models.geom_space import ClusterGeoms
from ulamdyn.unsup_models.traj_space import ClusterTrajs
from ulamdyn.unsup_models.dist_metrics import calc_rmsd
from ulamdyn.wrappers._auxiliary import *
from ulamdyn.nx_utils import *

__all__ = ["ClusteringAnalysis"]


class ClusteringAnalysis:
    @classmethod
    def _load_data(cls):
        # Step 1: Load XYZ data from all trajectories and align coordinates
        cls.gc = GetCoords()
        cls.gc.read_all_trajs()
        cls.rmsd_vals = cls.gc.rmsd

        # Step 2: build the dataset of properties that can be used for color map.
        cls.df_props = get_properties_data(cls.rmsd_vals)

    @classmethod
    def _load_params(cls, **kw):
        # List of valid keywords
        keywords = [
            "space",
            "method",
            "n_clusters",
            "descriptor",
            "mwc",
            "transform",
            "n_samples",
            "time_step",
            "data_scaler",
            "dist_metric",
            "n_cpus",
        ]
        for k in keywords:
            val = kw.get(k)
            setattr(cls, k, val)

        cls.space = cls.space.lower().strip()
        cls.method = cls.method.lower().strip().replace("-", "")
        if cls.method == "gmm":
            cls.method = "gaussian_mixture"
        cls.n_clusters = cls.n_clusters.lower().strip().replace(" ", "")
        cls.dist_metric = cls.dist_metric.lower().strip()

        if cls.n_clusters != "best":
            cls.n_clusters = list(map(int, cls.n_clusters.split(",")))
            if len(cls.n_clusters) == 1:
                cls.n_clusters = cls.n_clusters.pop()

        if cls.dist_metric == "rmsd":
            if cls.descriptor != "aXYZ":
                print("*********************************************************")
                print("WARNING:                                             \n")
                print("RMSD should be used only with the aXYZ descriptor.")
                print("The Euclidean distance will be used instead as default.")
                print("*********************************************************")
                cls.dist_metric = "euclidean"
            else:
                cls.dist_metric = calc_rmsd

        cls.stats_data = {}

    @classmethod
    def _save_props_data(cls, cluster_labels):
        if cls.space == "geoms":
            cls.df_props = cls.df_props.merge(
                cluster_labels, left_index=True, right_index=True, how="right"
            )
        elif cls.space == "trajs":
            cls.df_props = cls.df_props.merge(cluster_labels, on="TRAJ")
        csv_name = cls.space + "_" + cls.method + "_k" + str(cls.n_clusters)
        csv_name += "_properties.csv"
        cls.df_props.to_csv(csv_name, header=True, index=True, index_label="index")

    @classmethod
    def _save_cluster_geoms(cls, cluster_labels):
        cls.gc.build_dataframe()
        atom_labels = cls.gc.labels
        n_atoms = len(atom_labels)
        col_labels = [s for s in cluster_labels.columns if "labels" in s][0]
        df_xyz = cls.gc.dataset.copy()
        if cls.space == "geoms":
            df_xyz = df_xyz.merge(
                cluster_labels, left_index=True, right_index=True, how="right"
            )

            info2xyz = ["State"]
            if "RMSD" in cls.df_props.columns:
                info2xyz.append("RMSD")

            coords_cols = df_xyz.filter(regex="^x|^y|^z").columns.tolist()

            print("Saving geometries for each cluster...\n")
            for cluster in range(cls.n_clusters):
                select_cluster = col_labels + "==" + str(cluster)
                df_temp = df_xyz.query(select_cluster).sort_values(by=["time"])
                df_temp = df_temp[coords_cols]
                n_geoms = df_temp.shape[0]
                xyz_cluster = df_temp.values.reshape(n_geoms, n_atoms, 3)
                df_props_cluster = cls.df_props.query(select_cluster).sort_values(
                    by=["time"]
                )
                df_props_cluster = df_props_cluster.reset_index(drop=True)
                geoms = Geometries(atom_labels, df_props_cluster, info2xyz)
                out_name = cls.method + "_geoms_cluster" + str(cluster + 1) + ".xyz"
                geoms.save_xyz(xyz_cluster, out_name=out_name)

            print(
                "Saving average geometries corresponding to the clusters' centroids...\n"
            )
            select_cols = coords_cols + [col_labels]
            df_xyz = df_xyz[select_cols]
            df_xyz_mean = df_xyz.groupby(col_labels).mean()
            xyz_mean = df_xyz_mean.values.reshape(cls.n_clusters, -1, 3)

            geoms = Geometries(atom_labels)
            out_name = cls.method + "_geoms_centroids.xyz"
            geoms.save_xyz(xyz_mean, out_name=out_name)

        elif cls.space == "trajs":
            df_xyz = df_xyz.merge(cluster_labels, on="TRAJ")
            df_xyz_mean = df_xyz.groupby([col_labels, "time"]).mean()
            df_xyz_mean = df_xyz_mean.filter(regex="x|y|z")
            xyz_mean = df_xyz_mean.values.reshape(cls.n_clusters, -1, n_atoms, 3)
            for cluster in range(cls.n_clusters):
                print(
                    "Saving geometries for the average trajectory of cluster {}...\n".format(
                        cluster
                    )
                )
                select_cluster = col_labels + "==" + str(cluster)
                df_props_cluster = cls.stats_data["prop"].query(select_cluster)
                df_props_cluster = df_props_cluster.reset_index(drop=True)
                geoms = Geometries(atom_labels, df_props_cluster)
                out_name = cls.method + "_geoms_avg_traj" + str(cluster + 1) + ".xyz"
                geoms.save_xyz(xyz_mean[cluster], out_name=out_name)

    @classmethod
    def _build_stats(cls, cluster_labels):
        try:
            df_zmt = pd.read_csv("all_geoms_zmatrix.csv")
        except FileNotFoundError:
            zmt = ZMatrix(cls.gc)
            df_zmt = zmt.build_descriptor()

        vars_to_group = [c for c in cluster_labels.columns if "_labels" in c]
        if cls.space == "geoms":
            df_zmt = df_zmt.merge(
                cluster_labels, left_index=True, right_index=True, how="right"
            )
        elif cls.space == "trajs":
            vars_to_group += ["time"]
            df_zmt.insert(0, "TRAJ", cls.df_props["TRAJ"].values)
            df_zmt.insert(1, "time", cls.df_props["time"].values)
            df_zmt = df_zmt.merge(cluster_labels, on="TRAJ")

        print("Creating statistics for properties data based on clusters...\n")
        df_stats_prop = aggregate_data(cls.df_props, vars_to_group)
        cls.stats_data["prop"] = df_stats_prop.reset_index()
        csv_name = cls.space + "_" + cls.method + "_k" + str(cls.n_clusters)
        csv_name += "_stats_properties.csv"
        df_stats_prop.to_csv(csv_name, header=True, index=True, index_label="index")

        print("Creating statistics for Z-Matrix data based on clusters...\n")
        df_stats_zmt = aggregate_data(df_zmt, vars_to_group)
        cls.stats_data["zmat"] = df_stats_zmt.reset_index()
        csv_name = csv_name.replace("properties", "zmatrix")
        df_stats_zmt.to_csv(csv_name, header=True, index=True, index_label="index")

    @classmethod
    def run(cls, **kw):
        cls._load_data()
        cls._load_params(**kw)
        # Step 3: create the dataset to perform the clustering analysis.
        df_input = build_descriptor(cls.descriptor, cls.mwc, cls.transform, cls.gc)

        # Step 4: create instance for the clustering method.
        if cls.space == "geoms":
            model = ClusterGeoms(
                data=df_input,
                dt=cls.time_step,
                n_samples=cls.n_samples,
                scaler=cls.data_scaler,
                n_cpus=cls.n_cpus,
            )
        elif cls.space == "trajs":
            model = ClusterTrajs(
                data=df_input,
                dt=cls.time_step,
                scaler=cls.data_scaler,
                n_cpus=cls.n_cpus,
            )
        else:
            print("--------------------------------------------------------")
            print("ERROR:                                             \n")
            print("Invalid value for space keyword, {}.".format(cls.space))
            print("Available options: geoms or trajs")
            print("--------------------------------------------------------")
            sys.exit(1)

        try:
            df_labels = getattr(model, cls.method)(n_clusters=cls.n_clusters)
        except:
            print("--------------------------------------------------------")
            print("ERROR:                                             \n")
            print("Model {} not recognized or not implemented.".format(cls.method))
            print("Please select one of the available methods:")
            print("     geometries ---> K-Means, Hierarchical, Spectral or GMM.")
            print("   trajectories ---> K-Means.")
            print("--------------------------------------------------------")
            sys.exit(1)

        col_labels = [s for s in df_labels.columns if "labels" in s][0]
        cls.n_clusters = len(df_labels[col_labels].unique())

        print("Saving data set with all cluster labels...\n")
        filename = cls.space + "_" + cls.method + "_labels.csv"
        df_labels.to_csv(filename, header=True, index=True, index_label="index")

        cls._save_props_data(df_labels)
        cls._build_stats(df_labels)
        cls._save_cluster_geoms(df_labels)

        print("*" * 30)
        print("Clustering analysis finished!")
        print("*" * 30)
