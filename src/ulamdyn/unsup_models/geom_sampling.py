# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: May 17, 2023

import numpy as np
import pandas as pd

from ulamdyn.descriptors import R2
from ulamdyn.data_loader import GetProperties
from ulamdyn.data_writer import Geometries
from ulamdyn.unsup_models.geom_space import ClusterGeoms

__all__ = ["GeomSampling"]


class GeomSampling:
    """Class used to sample new molecular geometries from GMM clustering."""

    def __str__(self) -> str:
        """Provide a string representation of the class.

        :return: Short description of the class functionality.
        :rtype: str
        """
        return "Sample molecular geometries using Gaussian Mixture model."

    def __init__(
        self,
        descriptor,
        num_samples,
        n_clusters,
        transform=None,
        use_mwc=False,
        feat_scaler=None,
    ) -> None:
        self.use_mwc = use_mwc
        self.descriptor = descriptor
        self.transf_descr = transform
        self.n_samples = int(num_samples)
        self.scaler = feat_scaler
        self.n_clusters = n_clusters

        self.sampled_xyz = None
        self.descr_obj = None
        self.model = None
        self._properties_data = None

    def _build_descriptor(self):
        self.descr_obj = R2(use_mwc=self.use_mwc)
        df = self.descr_obj.build_descriptor(
            variant=self.descriptor, apply_to_delta=self.transf_descr
        )
        return df

    def _find_clusters(self, df):
        cluster = ClusterGeoms(data=df, n_samples=self.n_samples, scaler=self.scaler)
        df_labels = cluster.gaussian_mixture(n_clusters=self.n_clusters)
        self.model = cluster.model
        return df_labels

    @property
    def properties_data(self):
        return self._properties_data

    @properties_data.setter
    def properties_data(self, df_gmm_labels):
        gp = GetProperties()
        _ = gp.energies()
        _ = gp.populations()
        df_props = gp.dataset
        self._properties_data = df_props.merge(
            df_gmm_labels, left_index=True, right_index=True, how="right"
        )

    def _create_stats_data(self, cluster_id):
        df_stats = self.properties_data.groupby(by=["gmm_labels"])
        df_stats = df_stats.mean().reset_index()
        df_stats = df_stats.drop(["TRAJ"], axis=1)
        df_stats = df_stats.loc[cluster_id].to_frame().T
        n_samples = self.sampled_xyz.shape[0]
        df_stats = df_stats.loc[df_stats.index.repeat(n_samples)]
        df_stats = df_stats.reset_index(drop=True)
        df_stats["gmm_labels"] = df_stats["gmm_labels"].astype(int)
        return df_stats

    def gen_geometries(self, num_geoms, from_cluster=0, where_prop=""):
        if self.model is None:
            # STEP 1: Convert XYZ geometries into a descriptor (R2-like vector)
            df = self._build_descriptor()
            # STEP 2: Cluster geometries using Gaussian Mixture model and
            #         merge the cluster labels into the properties dataset
            self.properties_data = self._find_clusters(df)
        # STEP 3: Generate random feature vectors from the fitted Gaussian distribution
        if where_prop:
            df = self.properties_data.query(where_prop)
            from_cluster = df.groupby(by=["gmm_labels"]).size().idxmax()
        n_samples = num_geoms * self.n_clusters * 5
        new_samples, labels = self.model.sample(n_samples)
        # STEP 4: Select samples from specified cluster index
        filter = np.argwhere(labels == from_cluster).flatten()
        new_samples = new_samples[filter]
        new_samples = new_samples[:num_geoms]
        # STEP 5: Inverse transform the sampled R2 vectors and convert them to XYZ
        new_samples = self.descr_obj.inverse_transform(new_samples)
        new_xyz_geoms = self.descr_obj.reconstruct_xyz(new_samples)
        if self.use_mwc:
            new_xyz_geoms /= self.descr_obj.mass_weights
        # STEP 6: Align the sampled geometries to the reference one
        self.descr_obj.align_geoms(new_xyz_geoms)
        new_xyz_geoms = self.descr_obj.xyz
        self.sampled_xyz = new_xyz_geoms
        df_stats = self._create_stats_data(from_cluster)
        df_stats["RMSD"] = self.descr_obj.rmsd
        self._properties_data = df_stats
        return self.sampled_xyz

    def save_xyz(self, filename="sampled_geoms.xyz"):
        if (self.descr_obj.labels is not None) and (self.sampled_xyz is not None):
            atom_labels = self.descr_obj.labels
            add_info = [col for col in self.properties_data.columns if "DE" in col]
            add_info += ["RMSD"]
            geoms_obj = Geometries(atom_labels, self.properties_data, add_info)
            geoms_obj.save_xyz(self.sampled_xyz, filename)
        else:
            print(f"The sampled_xyz class variable is {self.sampled_xyz}.")
            print("There is no geometries available to save.")
