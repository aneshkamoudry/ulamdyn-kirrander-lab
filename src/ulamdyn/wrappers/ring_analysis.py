# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: June 6, 2022

# import sys

# import numpy as np

try:
    import modin.pandas as pd

except ModuleNotFoundError:
    import pandas as pd

from ulamdyn.data_loader import GetProperties
from ulamdyn.descriptors import RingParams
from ulamdyn.statistics import aggregate_data

__all__ = ["RingAnalysis"]


class RingAnalysis:
    atoms: str = None
    stats_by: str = None

    @classmethod
    def _load_params(cls, **kw):

        keywords = [
            "atoms",
            "stats_by",
        ]
        for k in keywords:
            val = kw.get(k)
            setattr(cls, k, val)

        try:
            cls.atom_indices = list(map(int, cls.atoms.split(",")))
        except ValueError:
            print("-----------------------------------------------------")
            print("Input value error: \n")
            print("The indices of the atoms composing the ring must be")
            print("provided as a list of comma separated integers.")
            print("-----------------------------------------------------\n")
            cls.atom_indices = None
            return

        cls.df = None
        cls.df_stats = None

    @classmethod
    def _build_stats(cls):
        if cls.stats_by is not None:
            vars_to_group = cls.stats_by.lower().split(",")
            if "state" in vars_to_group:
                gp = GetProperties()
                df_en = gp.energies()
                try:
                    cls.df.insert(2, "state", df_en["State"].values)
                except ValueError:
                    print(
                        "-----------------------------------------------------"
                    )
                    print("Mismatch error: \n")
                    print("The state labels from the energy data set cannot")
                    print("be inserted in the ring_params data set due to")
                    print("incompatible sizes.\n")
                    print("Size energy data = {}".format(df_en.shape[0]))
                    print("Size ring_params data = {}".format(cls.df.shape[0]))
                    print(
                        "-----------------------------------------------------"
                    )
                    return
            cls.df_stats = aggregate_data(cls.df, vars_to_group)
            cls.df_stats.to_csv(
                "stats_ring_params.csv", index=False, header=True
            )

    @classmethod
    def get_dataframes(cls):
        if cls.df is not None:
            return (cls.df, cls.df_stats)

    @classmethod
    def run(cls, **kw):
        cls._load_params(**kw)
        if cls.atom_indices is not None:
            ra = RingParams(ring_atom_ind=cls.atom_indices)
            cls.df = ra.build_dataframe(save_csv=True)
            if cls.df is not None and isinstance(cls.df, pd.DataFrame):
                cls._build_stats()
