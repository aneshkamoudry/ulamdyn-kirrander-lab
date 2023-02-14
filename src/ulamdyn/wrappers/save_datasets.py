# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: June 6, 2022

import sys

try:
    import modin.pandas as pd

except:
    import pandas as pd

from ulamdyn.data_loader import *
from ulamdyn.descriptors import *
from ulamdyn.statistics import aggregate_data
from ulamdyn.wrappers._auxiliary import *

__all__ = ["SaveDataset"]


class SaveDataset:
    @classmethod
    def _all(cls):
        print("Saving the full XYZ coordinates dataframe...\n")
        gc = GetCoords()
        gc.read_all_trajs()
        gc.save_csv
        print("Saving the R2 descriptor dataframe...\n")
        r2 = R2(gc)
        df = r2.build_descriptor(save_csv=True)
        print("Saving the Z-Matrix descriptor dataframe...\n")
        zmt = ZMatrix(gc)
        df = zmt.build_descriptor(save_csv=True)
        print("Saving the full properties dataframe...\n")
        df = get_properties_data()
        df.to_csv("all_properties.csv", index=False)

    @classmethod
    def _properties(cls):
        print("Saving the basic properties dataframe...\n")
        gp = GetProperties()
        properties = [
            "energies",
            "oscillator_strength",
            "populations",
        ]
        for prop in properties:
            _ = eval("gp." + prop + "()")
        df = gp.dataset
        df.to_csv("basic_properties.csv", index=False)

    @classmethod
    def _gradients(cls):
        print("Saving the XYZ gradients for each available state as dataframes...\n")
        gg = GetGradients()
        gg.build_dataframe(save_csv=True)

    @classmethod
    def _nacs(cls):
        print("Saving the NACs for each state pair as separated dataframes...\n")
        gnac = GetCouplings()
        gnac.build_dataframe(save_csv=True)

    @classmethod
    def _velocities(cls):
        print("Saving the atomic velocities for each MD step as a dataframe...\n")
        veloc = GetVelocities()
        veloc.build_dataframe(save_csv=True)

    @classmethod
    def _vibspec(cls):
        print("Saving the vibrational (power) spectra for all MD trajectories...\n")
        vs = VibrationalSpectra()
        df_spec = vs.build_dataframe()
        gp = GetProperties()
        df_prop = gp.energies()
        if df_spec.shape[0] == df_prop.shape[0]:
            df_spec.insert(1, "State", df_prop["State"].values)
        df_spec.to_csv("all_vibrational_spectra.csv", index=False, header=True)

    @classmethod
    def run(cls, data_to_save):
        cls.method = "_" + data_to_save.lower().strip()
        if cls.method in cls.__dict__:
            func = eval("cls." + cls.method)
            func()
        else:
            print("--------------------------------------------------------")
            print("ERROR:                                             \n")
            print("Invalid keyword, {}.".format(data_to_save))
            print("Available options: all, gradients, nacs or vibspec.")
            print("--------------------------------------------------------")
