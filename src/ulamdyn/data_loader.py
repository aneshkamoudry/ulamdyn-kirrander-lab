"""The :mod:`ulamdyn.data_loader` module is used to build datasets from SH output files."""

# Author: Aneshka Moudry <aneshkamoudry@gmail.com>
# Date: unknown

from __future__ import (
    absolute_import,
    annotations,
    division,
    print_function,
    unicode_literals,
    with_statement,
)

import os
from itertools import combinations, product
from typing import Tuple

import h5py
import numpy as np
import rmsd

try:
    import modin.pandas as pd

except ModuleNotFoundError:
    import pandas as pd

from ulamdyn.base import BaseClass
from ulamdyn.nx_utils import (
    BOHR_TO_ANG,
    HARTREE_TO_eV,
    check_nx_trajs,
    read_nx_control,
)

__all__ = ["GetCoords", "GetProperties"]

# %% Starting the first class: GetCoords
class GetCoords(BaseClass):
    """Read the Cartesian coordinates from Newton-X MD trajectories.

    .. note:: In the case of NX classical series, the Cartesian XYZ coordinates
              can be read either from dyn.out or dyn.xyz. For NX new series,
              the data will be read from the h5 file available in each TRAJ
              directory. Repeated geometries are skipped.

    It also provides a function to calculate the root-mean-squared deviation
    (RMSD) between each geometry read from the MD trajectories and a reference
    geometry. Before calculating the RMSD, the two geometries are aligned using
    the Kabsch algorithm (https://en.wikipedia.org/wiki/Kabsch_algorithm).

    This class does not require arguments in its constructor. All the outputs
    generate by the class are given in angstroms.

    Data attributes:
    ----------------
       ``trajectories`` (dict): contain indices of the trajectories available
                                with the respective maximum time to read.\n
       ``labels`` (numpy.ndarray): stores the sequence of atom labels.\n
       ``eq_xyz`` (numpy.ndarray): stores the XYZ matrix of the reference
                                   geometry (geom.xyz).\n
       ``xyz`` (numpy.ndarray): stores the XYZ matrices of all geometries read
                                from the TRAJ directories.\n
       ``rmsd`` (numpy.ndarray): vector of RMSD values between all geometries
                                 and the reference one.\n
       ``dataset`` (pandas.dataframe): stores a dataframe of the flattened
                                       XYZ matrices.
    """

    # Defining slots to optimize performance (RAM):
    __slots__ = [
        "trajectories",
        "labels",
        "eq_xyz",
        "xyz",
        "rmsd",
        "dataset",
        "traj_time",
    ]

    def __init__(self) -> None:
        """Class initialization."""
        # This variable contains a dictionary storing the available
        # trajectories as keys together with the corresponding t_max:
        # {'TRAJ1': tmax_1, 'TRAJ2': tmax_2,..., 'TRAJN': tmax_n}
        self.trajectories = check_nx_trajs()
        # Store the sequence of atom labels as a numpy array
        self.labels = None
        # Store the XYZ matrix of the reference geometry (np.array)
        self.eq_xyz = None
        # Try to read the reference geometry, geom.xyz file
        # If the file is available, the labels and eq_xyz variables
        # will be update with the data loaded by the function below.
        self.read_eq_geom()

        self.xyz = None
        self.rmsd = None
        self.dataset = None
        self.traj_time = None

    def __len__(self):
        if self.xyz is not None:
            return len(self.xyz)

    def __getitem__(self, idx) -> str:
        if self.xyz is not None:
            n_atoms = len(self.labels)
            geom_string = ""

            if isinstance(idx, int):
                traj, time = self.traj_time[idx]
                selected_geom = self.xyz[idx]
            elif isinstance(idx, tuple) and len(idx) == 2:
                traj, time = idx
                c1 = self.traj_time[:, 0] == traj
                c2 = self.traj_time[:, 1] == time
                selected_geom = self.xyz[(c1 & c2)][0]
            else:
                
                print("ERROR: Invalid input!")
                print("The input must be either a tuple with TRAJ and time")
                print("values or a single integer corresponding to the data")
                print("index.")
                return geom_string

            comment_line = f"TRAJ = {traj}  |  time = {time}"
            geom_string = str(n_atoms) + "\n" + comment_line + "\n"
            mask = "{:<6s} {:12.8f} {:12.8f} {:12.8f} \n"
            for label, atom_coords in zip(self.labels, selected_geom):
                geom_string += mask.format(label[0], *atom_coords)
            return geom_string

    def save_csv(self) -> None:
        """Save all loaded geometries (raw format) into a csv file.

        If the RMSD has been calculated, it will be included as an extra
        column in the XYZ coordinates data set.

        The default name of the output file is all_coordinates.csv.
        """
        if self.dataset is None:
            self.build_dataframe()

        df = self.dataset
        df.to_csv("all_coordinates.csv", index=False, header=True)

    def _insert_traj_time(self, df):
        traj_time_in_cols = all(
            col in df.columns.tolist() for col in ["TRAJ", "time"]
        )
        # Check first if the TRAJ and time columns already exist in dataframe
        if not traj_time_in_cols:
            if self.traj_time is not None:
                df.insert(0, "TRAJ", self.traj_time[:, 0])
                df.insert(1, "time", self.traj_time[:, 1])
                df["TRAJ"] = df["TRAJ"].astype(int)
        return df

    def build_dataframe(self) -> None:
        """Create a DataFrame containing flattened XYZ coordinates from all
        MD trajectories.

        After running this function, the class attribute
        :attr:`~ulamdyn.GetCoords.dataset` will be updated with the loaded
        DataFrame object.
        """
        if all(v is not None for v in [self.labels, self.xyz]):
            n_atoms = len(self.labels)
            col_names = [
                ["x" + str(i), "y" + str(i), "z" + str(i)]
                for i in range(1, n_atoms + 1)
            ]
            col_names = sum(col_names, [])
            print("Distance units: Angstrom")
            df = pd.DataFrame(
                self.xyz.reshape(-1, n_atoms * 3), columns=col_names
            )
            df = self._insert_traj_time(df)

            if self.rmsd is not None:
                df["RMSD"] = self.rmsd

            self.dataset = df

            print("\n-------------------------------------------------  ")
            print("  The size of the XYZ coordinates data set is\n   ")
            print("        Number of geometries = {}".format(df.shape[0]))
            print("          Number of features = {}".format(df.shape[1]))
            print("-------------------------------------------------  \n")

        else:
            print("---------------------------------------")
            print("The coordinates variable is empty!")
            print("There is no data available to save.")
            print("Please run the loader functions first.")
            print("---------------------------------------")

    @staticmethod
    def from_xyz(xyzfile) -> Tuple[np.ndarray, np.ndarray]:
        """Get XYZ coordinates from the out.xyz file.
        """
        count = 0
        xyz_geoms = []
        atom_labels = []

        with open(xyzfile, "r") as inp:
            for line in inp:
                vals = line.split()

                if len(vals) == 1 and count == 0:
                    num_atoms = int(vals[0])

                if len(vals) == 4:
                    coords = np.array(vals[-3:], dtype=np.float64)
                    coords = coords.reshape(-1, 3)
                    xyz_geoms.append(coords)
                    if count <= num_atoms + 2:
                        atom_labels.append(vals[0])

                count += 1

        atom_labels = np.asarray(atom_labels)
        xyz_array = np.vstack(xyz_geoms).reshape(-1, num_atoms, 3)

        return (atom_labels, xyz_array)
    
    @staticmethod
    def from_log(logfile) -> np.ndarray:
        """Get timestep array from the '/0/data/out.log' file of SH code.
        """
        time_lines = []
        with open(logfile, "r") as out:
            lines = out.read().split("\n")
            for line in lines:
                if "Time:" in line:
                    time_val = float(line.split()[1])
                    time_lines.append(time_val)   

        time_lines = np.array(time_lines)

        return time_lines

    def read_all_trajs(self, calc_rmsd: bool = True) -> None:
        """Concatenate the XYZ coordinates read from all MD trajectories.

        After running this method, the class attributes
        :attr:`~ulamdyn.GetCoords.labels` and :attr:`~ulamdyn.GetCoords.xyz`
        will be updated.
        """
        traj_id = []
        t_vals = []
        all_geoms = []

        # Instanciate the append before the loop for better efficiency
        append_trajs = traj_id.append
        append_times = t_vals.append
        append_all_geoms = all_geoms.append

        # Initiate the variable `ts` (timestamps) as an empty array to avoid
        # pylint error E606 (temporary solution).
        ts = np.array([])

        for trj in self.trajectories:
            tmax = self.trajectories.get(trj)
            results_dir = trj + "/0/data/"
            files = os.listdir(results_dir)
            if os.path.isfile(results_dir + "out.xyz"):
                xyzfile = results_dir + "out.xyz"
                logfile = results_dir + "out.log"
                atom_labels, xyz = self.from_xyz(xyzfile)
                ts = self.from_log(logfile)
                append_all_geoms(xyz)
                append_times(ts)
            else:
                print("\nSH output file not found.")
                print("Check the directory %s" % trj + "/0/data" + "\n")
                continue

            n_steps = len(ts)
            current_traj = np.int64(trj.replace("run_x", ""))
            append_trajs(np.full(n_steps, current_traj, dtype=np.int64))

        t_vals = np.concatenate(t_vals)
        traj_id = np.concatenate(traj_id)
        self.traj_time = np.concatenate(
            [traj_id.reshape(-1, 1), t_vals.reshape(-1, 1)], axis=1
        )
        self.xyz = np.concatenate(all_geoms, axis=0)
        # Important: capitalize the labels coming from h5 file (NX NS).
        self.labels = np.asarray([s.upper() for s in atom_labels])

        if calc_rmsd:
            self.align_geoms()

    def read_eq_geom(self) -> None:
        """Read the XYZ coordinates of a reference geometry.

        .. note:: This method should be executed before calculating the RMSD
                  with the :meth:`~ulamdyn.GetCoords.align_geoms` function.

        A file with name geom.xyz must be provided in the working directory
        (TRAJECTORIES). After reading the coordinates, the method will update
        the class attributes :attr:`~ulamdyn.GetCoords.labels` and
        :attr:`~ulamdyn.GetCoords.eq_xyz`.
        """
        try:
            atom_labels, ref_geom = self.from_xyz("geom.xyz")
            self.eq_xyz = np.squeeze(ref_geom, axis=0)
            self.labels = atom_labels
        except FileNotFoundError:
            print(
                "\n-----------------------------------------------------------"
            )
            print("Reference geometry not found!")
            print("The reference geometry is required to compute the RMSD. ")
            print(
                "Check if the geom.xyz file is available in the current dir."
            )
            print(
                "-----------------------------------------------------------\n"
            )

    def align_geoms(self, xyz_data=None) -> None:
        """Calculate the RMSD between the current and reference geometries.

        .. note:: Before calculating the RMSD, the method uses the Kabsch
                  algorithm to find the optimal alignment between the loaded
                  molecular geometry for each time step t and the reference
                  geometry.

        The calculated RMSD values will be stored in the class attribute
        :attr:`~ulamdyn.GetCoords.rmsd_values`, while the
        :attr:`~ulamdyn.GetCoords.xyz` attribute will be updated with the
        aligned geometries.
        """
        if self.eq_xyz is None:
            self.read_eq_geom()

        self.eq_xyz -= rmsd.centroid(self.eq_xyz)

        if xyz_data is None:
            if self.xyz is not None:
                xyz_data = self.xyz.copy()
            else:
                error_msg = (
                    "---------------------------------------------------"
                    + "\n "
                )
                error_msg += "XYZ coordinates not loaded!" + "\n "
                error_msg += (
                    "Please make sure that the read_all_trajs function "
                )
                error_msg += "has been executed." + "\n "
                error_msg += (
                    "---------------------------------------------------"
                    + "\n "
                )
                print(error_msg)
                return

        dim = xyz_data.shape
        aligned_geoms = np.empty(dim, dtype=np.float64)
        rmsd_values = np.empty((dim[0]), dtype=np.float64)

        for idx, geom in enumerate(xyz_data):
            geom -= rmsd.centroid(geom)
            U = rmsd.kabsch(geom, self.eq_xyz)
            geom = np.dot(geom, U)
            geom_rmsd = rmsd.rmsd(self.eq_xyz, geom)
            aligned_geoms[idx] = geom
            rmsd_values[idx] = geom_rmsd

        self.xyz = aligned_geoms
        self.rmsd = rmsd_values

# %% Starting the first class: GetProperties
class GetProperties(BaseClass):
    """Read all properties available in the JKSH MD trajectories.

    .. note:: This class does not require arguments in its constructor. All the
              energy quantities processed by the class are transformed from Ha
              to eV. For the other properties, the original units used in
              Newton-X are kept.

    Data attributes:
    ----------------
       ``trajectories`` (dict): trajectories ID (TRAJXX) found in the working
                                directory are stored as keys of the
                                dictionary, and the corresponding values are
                                the maximum simulation time to be read.\n
       ``dataset`` (pd.dataframe): store a dataframe with all properties.\n
       ``num_states`` (int): keep track of the number of states considered in
                             the MD simulation.\n

    """

    __slots__ = ["trajectories", "dataset", "num_states"]

    def __init__(self) -> None:
        """Class initializer."""
        # This variable contains a dictionary storing the available
        # trajectories as keys together with the corresponding t_max:
        # {'TRAJ1': tmax_1, 'TRAJ2': tmax_2,..., 'TRAJN': tmax_n}
        self.trajectories = check_nx_trajs()
        if self.trajectories:
            traj = list(self.trajectories)[0]
        # This class variable will be used to store a dataframe
        # with all properties read from the NX outputs
        self.dataset = None
        # Auxiliary variable to keep track of the number of states.
        # The default value will be updated in the energy function.

        # TO-DO: Create a function to read the number of states either from the
        # control.dyn file or from RESULTS/nx.log
        self.num_states = int(read_nx_control('run_x0221')['electronic']['states'][0])

    @property
    def save_csv(self) -> None:
        """Save the dataset with QM properties read from the JKSH trajectories.

        The following properties are included in the dataset:

        + trajectory index;
        + simulation time;
        + total energy;
        + energy gaps between states (eV);
        + oscillator strength (if available);
        + states population.
        + norm of the nonadiabatic coupling matrices (if available);
        + three highest MCSCF coefficients per state (only for NX/Columbus);
        """
        if self.dataset is not None:
            df = self.dataset.copy()
            df["time"] = df["time"].astype(object)

            print("                                             ")
            print("------------------------------------------------")
            print("  The size of the properties data set is\n  ")
            print("    Number of geometries = {}".format(df.shape[0]))
            print("     Number of features = {}".format(df.shape[1]))
            print("------------------------------------------------\n")

            df.to_csv(
                "all_properties.csv",
                index=False,
                header=True,
                float_format="%.10f",
            )

        else:
            print("---------------------------------------")
            print("The properties variable is empty!")
            print("There is no data to save.")
            print("Please run the loader functions first.")
            print("---------------------------------------\n")

    def _update_properties(self, df):
        # From now on, the input dataframe df must always contain the TRAJ and
        # time columns. These columns are necessary to identify/export the
        # differences between the two dfs.
        if self.dataset is None:
            print("\n-----------------------------------------------------")
            print("The properties dataset is empty.")
            print("Updating class variable with the current loaded data.")
            print("-----------------------------------------------------\n")
            self.dataset = df

        if all(isinstance(i, pd.DataFrame) for i in (self.dataset, df)):
            if self.dataset.shape[0] == df.shape[0]:
                current_cols = self.dataset.columns.tolist()
                cols_to_add = df.columns.difference(
                    self.dataset.columns
                ).tolist()
                if len(cols_to_add) != 0:
                    df = df[cols_to_add]
                    if "time" in current_cols:
                        dfs_to_merge = (self.dataset, df)
                    else:
                        dfs_to_merge = (df, self.dataset)
                    self.dataset = pd.concat(
                        dfs_to_merge, axis=1, ignore_index=False
                    )
            else:
                select_cols = ["TRAJ", "time"]
                df_diff = pd.concat(
                    [self.dataset[select_cols], df[select_cols]]
                ).drop_duplicates(keep=False)
                warning = "*************************************************\n"
                warning += (
                    "WARNING: The size of the dataframes does not match!\n\n"
                )
                warning += "    Dataset 1 contains {} rows\n".format(
                    self.dataset.shape[0]
                )
                warning += "    Dataset 2 contains {} rows\n\n".format(
                    df.shape[0]
                )
                warning += "Please check the mismatched TRAJ and time values\n"
                warning += "in the properties_diff.csv file.\n"
                warning += "*************************************************"
                print(warning)
                df_diff.to_csv("properties_diff.csv", index=False, header=True)
    
    def _energies_from_log(self) -> np.ndarray:
        all_energies = []
        append_energies = all_energies.append
        traj_id = []
        append_traj_id = traj_id.append

        for trj in self.trajectories:
            print("Reading energies from %s" % trj + "...")
            try:
                enfile = trj + "/0/data/out.dat"
                en = np.loadtxt(enfile, skiprows=1, usecols=range(11))
            except FileNotFoundError:
                print("\n-------------------------------------")
                print("Energy file not found.")
                print("Check the %s/0/data/ directory." % trj)
                print("-------------------------------------\n")
                continue

            tmax = self.trajectories.get(trj, 100000)
            mask = en[:, 0] <= tmax
            en = en[mask]
            append_energies(en)
            n_samples = en.shape[0]
            idx = np.int64(trj.replace("run_x", ""))
            append_traj_id(np.full(n_samples, idx, dtype=np.int64))

        traj_id = np.concatenate(traj_id)
        traj_id = traj_id.reshape(-1, 1)
        all_energies = np.concatenate(all_energies, axis=0)

        # After swapping the columns, the expected order of the columns should
        # be like:
        # ['time', 'potential energies', 'total energy', 'current state']
        #all_energies[:, -1], all_energies[:, -2] = (
        #    all_energies[:, -2],
        #    all_energies[:, -1].copy(),
        #)
        # And finally we add the column with the indices of the trajectories
        all_energies = np.concatenate((all_energies, traj_id), axis=1)

        return all_energies
    
    def energies(self):
        """Read / process the energy information from the out.dat file.

        :return: a processed dataset with the information of all trajectories
                 stacked, and containing the following columns "TRAJ", "time",
                 "State", "Total_Energy" plus the energy gaps between the
                 accessible states (e.g., DE12) and binary columns to identify
                 the hopping points (e.g., Hops_S21).
        :rtype: pandas.DataFrame | modin.pandas.dataframe.DataFrame
        """
        all_energies = self._energies_from_log()

        self.num_states = int(read_nx_control('run_x0221')['electronic']['states'][0])
        state_labels = ["S" + str(i) for i in range(self.num_states)]
        pop_labels = [f"population({state})" for state in state_labels]
        pe_labels = [f"PE({state})" for state in state_labels]

        col_names = (
            ["time", "Active_State"] +
            pop_labels +
            state_labels +
            ["Total_PE", "Total_KE", "Total_Energy", "TRAJ"]
        )
        df = pd.DataFrame(all_energies, columns=col_names)
        df["TRAJ"] = df["TRAJ"].astype(int)
        df["State"] = df["Active_State"].astype(int)

        # Auxiliary variable used to find the hopping points
        df["State_Next"] = df.groupby(by="TRAJ")["State"].shift(
            -1, fill_value=-10
        )

        
        # Loop to calculate energy difference between all possible pair of
        # states
        
        for i, j in combinations(range(self.num_states), 2):
            #from_to = j.replace("S", "") + i.replace("S", "")
            from_to = str(j) + str(i)
            new_col = "DE" + from_to


            state_i = "S" + str(i)  # Convert back to state label format (S1, S2, etc.)
            state_j = "S" + str(j) 
            df[new_col] = df[state_j] - df[state_i]
        
            si = i
            sj = j
            # Create a binary column to identify hopping geometries
            # The first condition corresponds to hoppings by state decay
            new_col = "Hops_S" + str(sj) + str(si)
            condition = (df["State"] == sj) & (df["State_Next"] == si)
            df[new_col] = np.where(condition, 1, 0)
            # The second condition takes into account upward hoppings
            new_col = "Hops_S" + str(si) + str(sj)
            condition = (df["State"] == si) & (df["State_Next"] == sj)
            df[new_col] = np.where(condition, 1, 0)
        
        cols_to_drop = state_labels[1:] + ["Active_State", "State_Next", "Total_KE", "Total_PE"] + pop_labels 
        df.drop(cols_to_drop, axis=1, inplace=True)

        # Remove all columns that contains only zeros
        df = df.loc[:, (df != 0).any(axis=0)]

        all_cols = df.columns.tolist()
        base_cols = ["TRAJ", "time", "State", "Total_Energy"]
        cols_reordered = base_cols + list(set(all_cols) - set(base_cols))
        df = df[cols_reordered]

        self._update_properties(df)
        
        return df