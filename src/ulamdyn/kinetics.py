## Author: Max Pinheiro Jr <maxjr82@gmail.com>
## Date: March 10 2021
from __future__ import (absolute_import, division, print_function,
                        unicode_literals, with_statement)

import os
import sys
import glob
import numpy as np

try:
    import modin.pandas as pd
    import ray
    ray.init()
except:
    import pandas as pd

from itertools import combinations
from ulamdyn.utilities import *

BOHR_TO_ANG = 0.529177210903
HARTREE_TO_KCAL = 627.5096080305927
HARTREE_TO_eV = 27.211399
PROTON_MASS = 1822.888515

__all__ = ["GetVelocities", "KineticEnergy"]

#%% Starting the first class: GetCoords
class GetVelocities:
    """Collect velocities from all MD trajectories of Newton-X.

    """

    # Defining slots to optimize performance (RAM):
    __slots__ = ['trajectories', 'n_atoms', 'veloc']

    def __str__(self):
         return "Data handler class for NX-MD velocities."

    def __init__(self, n_atoms=None):
        # This variable contains a list of all available trajectories:
        # [TRAJ1, TRAJ2,..., TRAJN]
        self.trajectories = get_traj_dirs()

        if n_atoms == None:
            self.n_atoms = get_num_atoms()
        else:
            self.n_atoms = int(n_atoms)

        self.veloc = None    

    def from_dyn(self, outfile):
        """Reads the velocities from one trajectory (dyn.out file) of Newton-X.

        """

        read_veloc = False
        n_atoms = self.n_atoms

        f = open(outfile, 'r')
        lines = f.read()

        if "velocity" not in lines:
            f.close()
            print("\nThe %s file does not contain information of velocities." % outfile)
            return
        else:    
            lines = lines.split('\n')

            veloc_lines = list(filter(lambda k: 'velocity' in k, lines))
            n_rows = len(veloc_lines)
            veloc = np.empty((n_rows, n_atoms, 3), dtype=np.float64)
            count_atoms = 0
            count_steps = -1

            for line in lines:

                if read_veloc:
                    vals = line.split()
                    if len(vals) == 3:
                        if count_atoms == n_atoms:
                            count_atoms = 0
                            read_veloc = False
                            continue
                        veloc[count_steps][count_atoms] = np.array(vals, dtype=np.float64)
                        count_atoms += 1
                    else:
                        count_atoms = 0
                        read_veloc = False

                if 'velocity' in line:
                    read_veloc = True
                    count_steps += 1

        f.close()

        return veloc

    def read_all_trajs(self):
        """Load the atom velocities from all available trajectories, and 
        store the stacked matrices in the class variable ``veloc``.

        """ 
        all_veloc = list()
        for trj in self.trajectories:
            print("Reading velocities from %s" % trj + "...")
            if os.path.isfile(trj + '/RESULTS/dyn.out'):
                dynfile = trj + '/RESULTS/dyn.out'
                veloc = self.from_dyn(dynfile)
                all_veloc.append(veloc)
            else:
                print ("\nFile dyn.out not found.") 
                print("Check the directory %s" % trj + "/RESULTS" + "\n")
        
        all_veloc = np.concatenate(all_veloc, axis = 0)
        self.veloc = all_veloc

    @classmethod
    def from_all_trajs(cls, n_atoms=None):
        instance = cls(n_atoms)
        instance.read_all_trajs()
        veloc = instance.veloc
        return veloc    

class KineticEnergy:

    # Defining slots to optimize performance (RAM):
    __slots__ = ['trajectories', 'n_atoms', 'energies', 'atom_mass', 'atom_labels']

    def __str__(self):
         return "Kinetic energy calculator."

    def __init__(self, n_atoms=None):
        
        self.trajectories = get_traj_dirs()
        if n_atoms == None:
            self.n_atoms = get_num_atoms()
        else:
            self.n_atoms = int(n_atoms)

        self.energies = None
        self.atom_labels, self.atom_mass = self._get_labels_masses

    @property
    def _get_labels_masses(self):

        traj = self.trajectories[0]
        nx_geom_file = traj + '/geom'
        try:
            data = np.loadtxt(nx_geom_file, usecols=(0,5), unpack=True, 
                              dtype={'names': ('labels', 'masses'), 
                                     'formats': ('U1', 'float')})
            atom_labels, atom_mass = data
            atom_labels = atom_labels[:self.n_atoms]
            atom_mass = atom_mass[:self.n_atoms] * PROTON_MASS

        except:
            print("--------------------------------------------------------------")
            print("ERROR:                                                        ")
            print("geom file not found or n_atoms is larger than atom_mass array.") 
            print("Check the content of TRAJ%s directory" % traj                  )
            print("--------------------------------------------------------------")

        return (atom_labels, atom_mass)

    @staticmethod
    def _atom_speed(veloc_xyz: np.ndarray):
        """Computes the speed of every atom in the molecular system.

        The input should be given a matrix with the atomic velocities read from
        all trajectories. Shape = (Nsteps, Natoms, 3).
        """
        return np.linalg.norm(veloc_xyz, axis = 2)

    def calc_per_atom(self):
        """Calculates the kinetic energy of each atom for all MD trajectories available.

        """

        all_veloc = GetVelocities.from_all_trajs(self.n_atoms)
        all_speeds = self._atom_speed(all_veloc)
        v_squared = all_speeds**2
        all_ekin_atoms = 0.5 * np.multiply(v_squared, self.atom_mass)
        all_ekin_atoms *= HARTREE_TO_eV

        return all_ekin_atoms

    def calc_per_molecule(self, n_mols_per_type, n_atoms_per_mol):
        """Sum the atomic contributions of kinetic energy for each molecule.

        .. note:: In the current version, this function only supports two different types 
                  of molecules in the geom file, where the first one should always 
                  correspond to the solute. The function will be generalized in future
                  versions to support any number of different molecules.

        """

        n_mols1, n_mols2 = n_mols_per_type
        n_atoms1, n_atoms2 = n_atoms_per_mol

        all_ekin_atoms = self.calc_per_atom()
        # Here we assume that the first n_atoms1 in the geom file of NX
        # corresponds to the solute molecule. Therefore, we slice the 
        # all_ekin_atoms matrix using the n_atoms1 variable.
        ekin_solute = all_ekin_atoms[:, :n_atoms1]
        ekin_solute = ekin_solute.sum(axis=1, keepdims=True)
        ekin_solvent = all_ekin_atoms[:, n_atoms1:]
        ekin_solvent = ekin_solvent.reshape(-1,n_mols2,n_atoms2).sum(axis=2)
        ekin_molecules = np.concatenate((ekin_solute, ekin_solvent), axis=1)

        return ekin_molecules

    def build_dataframe(self, discretization_level='atom', n_mols_per_type=None, 
                        n_atoms_per_mol=None, save_csv=False):
        """Creates a pandas DataFrame containing the atomic kinetic energies.

        """

        if discretization_level == 'atom':
            all_ekin = self.calc_per_atom()
            atom_id = np.arange(1,len(self.atom_labels)+1)
            col_names = ['Ekin_'+label+str(n) for label, n in zip(self.atom_labels,atom_id)]
        elif discretization_level == 'molecule':
            if None not in [n_mols_per_type,n_atoms_per_mol]:
                all_ekin = self.calc_per_molecule(n_mols_per_type, n_atoms_per_mol)
                n_mols = all_ekin.shape[1]
                col_names = ['Ekin_mol'+str(n+1) for n in range(n_mols)]
            else:
                print("--------------------------------------------------------------")
                print("ERROR:"                                                        )
                print("A list with the number of molecules per type, and the"         ) 
                print("corresponding number of atoms per different molecule"          )
                print("must be provided to calculate the kinetic energy per molecule" )
                print("--------------------------------------------------------------")
                return

        df = pd.DataFrame(all_ekin, columns=col_names)

        if save_csv:
            df.to_csv('all_kinetic_energies.csv', index=False, header=True)

        return df