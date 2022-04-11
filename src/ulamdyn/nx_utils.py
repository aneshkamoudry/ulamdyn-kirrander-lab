"""Auxiliary functions and constants used by the main modules."""
# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: May 17 2021
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
import numpy as np

#%% List of constants
BOHR_TO_ANG = 0.529177210903
HARTREE_TO_KCAL = 627.5096080305927
HARTREE_TO_eV = 27.211399
PROTON_MASS = 1822.888515

#%% Auxiliary functions

# This function is used to return one list with all 'TRAJXX' directories
# sorted in ascending order.
def get_traj_dirs():
    dirs_list = glob.glob("TRAJ[0-9]*")
    dirs_list = sorted(dirs_list, key=lambda x: int(x.rsplit("TRAJ")[1]))
    return dirs_list


def get_nx_version(traj_dir):
    nx_version = None
    if os.path.isdir(traj_dir):
        if os.path.isfile(traj_dir + "/" + "configuration.inp"):
            nx_version = "ns"
        elif os.path.isfile(traj_dir + "/" + "control.dyn"):
            nx_version = "cs"
        else:
            print("\nERROR: Newton-X version not recognized!\n")
    else:
        print("The {} directory does not exist.".format(traj_dir))
    return nx_version


def get_labels_masses(traj_dir, n_atoms=None):

    if os.path.isfile(traj_dir + "/geom.orig"):
        # This is the file name for new Newton-X
        nx_geom_file = traj_dir + "/geom.orig"
    else:
        # This is the standard geom input file in classical Newton-X
        nx_geom_file = traj_dir + "/geom"
    try:
        data = np.loadtxt(
            nx_geom_file,
            usecols=(0, 5),
            unpack=True,
            dtype={"names": ("labels", "masses"), "formats": ("U1", "float")},
        )
        atom_labels, atom_mass = data
        atom_mass *= PROTON_MASS
    except Exception as e:
        print("--------------------------------------------------------------")
        print("ERROR:                                                        ")
        print("geom file not found.")
        print("Check the content of %s directory" % traj_dir)
        print("--------------------------------------------------------------")

    if n_atoms:
        atom_labels = atom_labels[:n_atoms]
        atom_mass = atom_mass[:n_atoms]

    return (atom_labels, atom_mass)


def read_nx_control(traj_dir):

    nx_version = get_nx_version(traj_dir)
    if nx_version == "cs":
        control_input = traj_dir + "/control.dyn"
    elif nx_version == "ns":
        control_input = traj_dir + "/configuration.inp"
    else:
        print("\nERROR: Newton-X version not recognized!\n")
        return {}

    control = {}
    with open(control_input, "r") as nxinp:
        for line in nxinp:
            if len(line.split()) >= 3:
                keyword = line.split()[0]
                value = line.split("=")[1].split()[0]
                try:
                    control[keyword] = float(value)
                except ValueError:
                    control[keyword] = value

    return control


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


def check_nx_trajs():
    traj_tmax = {}
    traj_list = get_traj_dirs()
    if len(traj_list) == 0:
        return traj_tmax

    if os.path.isfile("trajs_tmax.dat"):
        lines = open("trajs_tmax.dat", "r").readlines()
        for line in lines:
            traj = line.split()[0].strip()
            tmax = float(line.split()[1])
            if traj in traj_list:
                traj_tmax[traj] = tmax
    else:
        nx_version = get_nx_version(traj_list[0])
        if nx_version == "cs":
            for traj in traj_list:
                try:
                    nxlog = traj + "/RESULTS/nx.log"
                    f = open(nxlog, "r")
                except FileNotFoundError:
                    print("\n---------------------------------------")
                    print("nx.log file not found.")
                    print("Check the %s/RESULTS directory." % traj)
                    print("---------------------------------------\n")
                    continue

                lines = f.read().split("\n")
                t_last = list(filter(lambda k: "TIME" in k, lines))[-3:]
                end_flag = "NEWTON-X ends here"
                normal_run = any(end_flag in line for line in lines[-5:])
                step = -1 if normal_run else -2
                tmax = float(t_last[step].split()[4])
                traj_tmax[traj] = tmax
        else:
            config = read_nx_control(traj_list[0])
            tmax = config.get("tmax", 10000)
            traj_tmax = dict.fromkeys(traj_list, tmax)

    return traj_tmax
