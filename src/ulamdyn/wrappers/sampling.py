# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: May 29, 2023

import sys
from ulamdyn.unsup_models.geom_sampling import GeomSampling

__all__ = ["SampleGeometries"]


class SampleGeometries:
    @classmethod
    def _load_params(cls, **kw):
        # List of valid keywords
        keywords = [
            "n_clusters",
            "descriptor",
            "mwc",
            "transform",
            "n_samples",
            "time_step",
            "data_scaler",
            "n_new_geoms",
            "sample_from",
        ]
        for k in keywords:
            val = kw.get(k)
            setattr(cls, k, val)

        if "R2" not in cls.descriptor:
            print("Geometry sampling is available only for R2-type descriptors.")
            sys.exit()

        cls.n_samples = int(cls.n_samples)
        cls.n_new_geoms = int(cls.n_new_geoms)
        cls.sample_from = cls.sample_from.strip()
        cls.n_clusters = cls.n_clusters.lower().strip().replace(" ", "")
        cls.n_clusters = int(cls.n_clusters)

    @classmethod
    def run(cls, **kw):
        cls._load_params(**kw)
        gs = GeomSampling(
            descriptor=cls.descriptor,
            n_samples=cls.n_samples,
            n_clusters=cls.n_clusters,
            transform=cls.transform,
            use_mwc=cls.mwc,
            feat_scaler=None,
        )
        if cls.sample_from.isdigit():
            from_cluster = int(cls.sample_from)
            where_prop = ""
        else:
            from_cluster = 0
            where_prop = cls.sample_from
        new_geoms = gs.gen_geometries(
            num_geoms=cls.n_new_geoms, from_cluster=from_cluster, where_prop=where_prop
        )
        gs.save_xyz()
