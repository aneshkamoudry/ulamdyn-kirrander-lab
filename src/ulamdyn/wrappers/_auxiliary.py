# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: June 3, 2021
import os
import sys
import numpy as np

try:
    import modin.pandas as pd

except:
    import pandas as pd

from ulamdyn.data_loader import *
from ulamdyn.kinetics import *
from ulamdyn.descriptors import *
from ulamdyn.statistics import *
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


def build_descriptor(descriptor, mwc, transform, getcoords_obj):
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
        r2 = R2(getcoords_obj, mwc)
        df_r2 = r2.build_descriptor(variant=descriptor, apply_to_delta=transform)
        df_r2.to_csv(descriptor + ".csv", index=False)
        return df_r2
    elif descriptor in ["Zmat", "delta-Zmat"]:
        zmt = ZMatrix(getcoords_obj)
        dfs_dict = {
            "Zmat": zmt.build_descriptor(),
            "delta-Zmat": zmt.build_descriptor(delta=True, apply_to_delta=transform),
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
