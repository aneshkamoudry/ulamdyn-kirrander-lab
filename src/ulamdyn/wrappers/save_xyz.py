# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: June 6, 2022

import numpy as np

try:
    import modin.pandas as pd

except:
    import pandas as pd

from ulamdyn.data_loader import *
from ulamdyn.data_writer import Geometries
from ulamdyn.wrappers._auxiliary import *
from ulamdyn.nx_utils import *

__all__ = ["SaveXYZ"]

class SaveXYZ:
    @classmethod
    def _load_params(cls, **kw):
        keywords = [
            "save_xyz",
            "use_au",
        ]
        for k in keywords:
            val = kw.get(k)
            setattr(cls, k, val)

        cls.save_xyz = cls.save_xyz.strip().split(",")
        cls.data_type = "_" + cls.save_xyz[0]
        if len(cls.save_xyz) == 1:
            cls.select_points = ["all"]
        elif len(cls.save_xyz) > 1:
            cls.select_points = cls.save_xyz[1:]

    @classmethod
    def _load_geoms(cls):
        cls.gc = GetCoords()
        cls.gc.read_all_trajs(calc_rmsd=False)
        cls.rmsd_vals = cls.gc.rmsd
        cls.labels = cls.gc.labels
        cls.n_atoms = len(cls.labels)
        cls.all_geoms = cls.gc.xyz.copy()

    @classmethod
    def _load_grads(cls):
        cls.gg = GetGradients()
        cls.gg.build_dataframe()
        cls.all_grads = cls.gg.all_grads.copy()

    @classmethod
    def _load_properties(cls):
        if not hasattr(cls, "rmsd_vals"):
            cls.rmsd_vals = None
        cls.df_props = get_properties_data(cls.rmsd_vals)
        cls.df_props = cls.df_props.round(4)

    @classmethod
    def _select_hops(cls):
        col_hoppings = [col for col in cls.df_props.columns if "Hops" in col]
        if len(cls.select_points) > 1:
            col_hoppings = ["Hops_" + cls.select_points[1].capitalize()]

        for col in col_hoppings:
            indices = cls.df_props[cls.df_props[col] == 1].index.tolist()
            if len(indices) > 0:
                df_props_hops = (
                    cls.df_props[cls.df_props[col] == 1].copy().reset_index(drop=True)
                )
                hopping_geoms = cls.all_geoms[indices].copy()
                states_pair = col.split("_")[1]
                egap_info = [states_pair.replace("S", "DE")]
                out_name = "Geoms_Hopping_" + states_pair + ".xyz"
                geoms = Geometries(cls.labels, df_props_hops, cls.info2xyz + egap_info)
                geoms.save_xyz(hopping_geoms, out_name)

    @classmethod
    def _filter_by_property(cls):
        mask = cls.select_points[0]
        cls.df_props = cls.df_props.query(mask)
        indices = cls.df_props.index.tolist()
        if hasattr(cls, "all_geoms"):
            cls.all_geoms = cls.all_geoms[indices]
        elif hasattr(cls, "all_grads"):
            for state in cls.all_grads.keys():
                cls.all_grads[state] = cls.all_grads[state][indices]
        cls.df_props = cls.df_props.reset_index(drop=True)

    @classmethod
    def _geoms(cls):
        cls._load_geoms()
        cls._load_properties()
        units = 'bohr' if cls.use_au else 'Å'

        if cls.use_au:
            cls.all_geoms /= BOHR_TO_ANG

        cls.info2xyz = []
        if "RMSD" in cls.df_props.columns:
            cls.info2xyz.append("RMSD")

        if cls.select_points[0].lower() == "hops":
            print(f"Writing hopping geometries (in {units}) to XYZ file...\n")
            cls._select_hops()
        else:
            outname = "all_geometries.xyz"
            if cls.select_points[0] != "all":
                print(f"Writing selected geometries (in {units}) to XYZ file...\n")
                cls._filter_by_property()
                outname = "selected_geometries.xyz"
            print(f"Writing all molecular geometries (in {units}) to XYZ file...\n")
            geoms = Geometries(cls.labels, cls.df_props, cls.info2xyz)
            geoms.save_xyz(cls.all_geoms, outname)

    @classmethod
    def _grads(cls):
        cls._load_grads()
        cls._load_properties()
        units = 'Ha/bohr' if cls.use_au else 'Ha/Å'

        if cls.select_points[0] != "all":
            cls._filter_by_property()

        for state in cls.all_grads.keys():
            print(
                "Writing gradient matrices (in {}) of state {} to XYZ file...\n".format(units,state)
            )
            grads = cls.all_grads[state].copy()
            grads *= 1 / HARTREE_TO_eV
            n_atoms = grads.shape[1]
            empty_labels = np.array([[""] for i in range(n_atoms)])
            if cls.use_au:
                grads *= BOHR_TO_ANG
            out_name = "all_gradients_" + str(state).lower() + ".xyz"
            geoms = Geometries(empty_labels, cls.df_props)
            geoms.save_xyz(grads, out_name)

    @classmethod
    def run(cls, **kw):
        cls._load_params(**kw)
        if cls.data_type in cls.__dict__:
            func = eval("cls." + cls.data_type)
            func()
        else:
            print("--------------------------------------------------------")
            print("ERROR:                                             \n")
            print("Invalid keyword, {}.".format(cls.save_xyz[0]))
            print("The input should start with geoms or grads.")
            print("--------------------------------------------------------")
