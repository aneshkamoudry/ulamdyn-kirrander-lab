# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: May 29, 2023

import sys

from ulamdyn.unsup_models.geom_sampling import GeomSampling

__all__ = ["SampleGeometries"]


class SampleGeometries:
    # Declare attributes with default values
    descriptor: str = None
    n_samples: int = None
    mwc: bool = None
    transform: str = None
    time_step: float = None
    data_scaler: str = None
    n_new_geoms: int = None
    sample_from: str = None
    n_clusters: int = None

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

        if cls.descriptor is None or "R2" not in str(cls.descriptor):
            err_msg = "Geometry sampling is available only for "
            err_msg += "R2-type descriptors."
            raise ValueError(err_msg)

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

        _ = gs.gen_geometries(
            num_geoms=cls.n_new_geoms,
            from_cluster=from_cluster,
            where_prop=where_prop,
        )
        gs.save_xyz()
