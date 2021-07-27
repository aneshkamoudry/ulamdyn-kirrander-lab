## Author: Max Pinheiro Jr <maxjr82@gmail.com>
## Date: May 17 2021
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
    with_statement,
)

import os
import sys
import glob
import h5py

#%% List of constants
BOHR_TO_ANG = 0.529177210903
HARTREE_TO_KCAL = 627.5096080305927
HARTREE_TO_eV = 27.211399

#%% Auxiliary functions

# This function is used to return one list with all 'TRAJXX' directories
# sorted in ascending order.
def get_traj_dirs():
    dirs_list = glob.glob("TRAJ[0-9]*")
    dirs_list = sorted(dirs_list, key=lambda x: int(x.rsplit("TRAJ")[1]))
    return dirs_list


def get_nx_version(traj_dir):
    if os.path.isdir(traj_dir):
        if os.path.isfile(traj_dir + "/" + "configuration.inp"):
            nx_version = "ns"
        elif os.path.isfile(traj_dir + "/" + "control.dyn"):
            nx_version = "cs"
        else:
            print("ERROR:")
            print("Newton-X version not recognized!")
            sys.exit()
    else:
        print("The {} directory does not exist.".format(traj_dir))
    return nx_version


def read_h5_nx(traj):
    files = os.listdir(traj + "/RESULTS")
    h5file = next((f for f in files if f.endswith(".h5")), None)
    if h5file:
        h5file = traj + "/RESULTS/" + h5file
        data = h5py.File(h5file, "r")
    else:
        print("\n-------------------------------------")
        print("H5 data file not found.")
        print("Check the %s/RESULTS directory." % traj)
        print("-------------------------------------\n")
        return None

    return data


def get_num_atoms():
    try:
        n_atoms = int(open("geom.xyz").readline().rstrip())
        return n_atoms
    except FileNotFoundError:
        print("\n-----------------------------------------------------------")
        print("Reference geometry not found!")
        print("Check if the geom.xyz file is available in the current dir.")
        print("-----------------------------------------------------------\n")
