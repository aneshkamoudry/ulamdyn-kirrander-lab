# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: June 10, 2022

import sys

from ulamdyn.data_loader import GetCoords
from ulamdyn.unsup_models.dist_metrics import calc_rmsd
from ulamdyn.unsup_models.geom_space import DimensionReduction
from ulamdyn.wrappers._auxiliary import build_descriptor, get_properties_data

__all__ = ["DimensionReductionAnalysis"]


class DimensionReductionAnalysis:
    descriptor: str = None
    mwc: bool = None
    transform: str = None
    n_samples: int = None
    time_step: float = None
    data_scaler: str = None
    method: str = None
    n_dim: int = None
    dist_metric: str = None
    # Parameter specific to KPCA or Isomap
    kernel: str = None
    # Parameter specific to t-SNE
    perplexity: float = None
    n_cpus: int = None

    @classmethod
    def _load_data(cls):
        # Step 1: Load XYZ data from all trajectories and align coordinates
        cls.gc = GetCoords()
        cls.gc.read_all_trajs()
        cls.rmsd_vals = cls.gc.rmsd

        # Step 2: build the dataset of properties that can be used for
        #         color map.
        cls.df_props = get_properties_data(cls.rmsd_vals)

    @classmethod
    def _load_params(cls, **kw):

        keywords = [
            "descriptor",
            "mwc",
            "transform",
            "n_samples",
            "time_step",
            "data_scaler",
            "method",
            "n_dim",
            "dist_metric",
            "kernel",
            "perplexity",
            "n_cpus",
        ]
        for k in keywords:
            val = kw.get(k)
            setattr(cls, k, val)

        cls.method = cls.method.lower().strip().replace("-", "")

        if cls.dist_metric == "rmsd":
            if cls.descriptor != "aXYZ":
                print(
                    "*********************************************************"
                )
                print(
                    "WARNING:                                             \n"
                )
                print("RMSD should be used only with the aXYZ descriptor.")
                print(
                    "The Euclidean distance will be used instead as default."
                )
                print(
                    "*********************************************************"
                )
                cls.dist_metric = "euclidean"
            else:
                cls.dist_metric = calc_rmsd

    @classmethod
    def _save_props_data(cls, reduced_data):
        print("Saving the merged (reduced + properties) data sets...\n")
        reduced_data = reduced_data.merge(
            cls.df_props, left_index=True, right_index=True, how="left"
        )
        csv_name = cls.method + "_ndim" + str(cls.n_dim) + "_"
        csv_name += cls.descriptor.lower() + ".csv"
        reduced_data.to_csv(
            csv_name, header=True, index=True, index_label="index"
        )

    @classmethod
    def run(cls, **kw):
        cls._load_data()
        cls._load_params(**kw)

        # Step 3: create the dataset to perform dimension reduction analysis
        df_input = build_descriptor(
            cls.descriptor, cls.mwc, cls.transform, cls.gc
        )

        # Step 4: create an instance of the dimensionality reduction class
        dimred = DimensionReduction(
            data=df_input,
            dt=cls.time_step,
            n_samples=cls.n_samples,
            scaler=cls.data_scaler,
            n_cpus=cls.n_cpus,
        )

        # Step 5: check for the available models and run the calculation
        if cls.method == "pca":
            df_reduced = dimred.pca(n_components=cls.n_dim, calc_error=True)
        elif cls.method == "kpca":
            df_reduced = dimred.kpca(n_components=cls.n_dim, kernel=cls.kernel)
        elif cls.method == "isomap":
            df_reduced = dimred.isomap(
                n_components=cls.n_dim, metric=cls.dist_metric, calc_error=True
            )
        elif cls.method == "tsne":
            df_reduced = dimred.tsne(
                n_components=cls.n_dim,
                metric=cls.dist_metric,
                perplexity=cls.perplexity,
            )
        else:
            print("--------------------------------------------------------")
            print("ERROR:                                             \n")
            print("Model type not recognized or not implemented!")
            print("Please select one of the available methods:")
            print("PCA, KPCA, Isomap or t-SNE.")
            print("--------------------------------------------------------")
            sys.exit(1)

        cls._save_props_data(df_reduced)

        print("*" * 45)
        print("Dimensionality reduction analysis finished!")
        print("*" * 45)
