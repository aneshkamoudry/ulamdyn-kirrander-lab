## Author: Max Pinheiro Jr <maxjr82@gmail.com>
## Date: 03/10/2021

from __future__ import (absolute_import, division, print_function,
                        unicode_literals, with_statement)

import os
import sys
import glob
import rmsd
import numpy as np

try:
    import modin.pandas as pd
    import ray
    ray.init()
except:
    import pandas as pd

from itertools import combinations

filedir = os.path.dirname(__file__)

BOHR_TO_ANG = 0.529177210903
HARTREE_TO_KCAL = 627.5096080305927
HARTREE_TO_eV = 27.211399

# This function is used to return one list with all 'TRAJXX' directories
# sorted in ascending order.
def get_traj_dirs():
    dirs_list = glob.glob("TRAJ[0-9]*")
    dirs_list = sorted(dirs_list, key = lambda x: int(x.rsplit("TRAJ")[1]))
    return dirs_list

class GetCoords:
    """
    Class object used to read the Cartesian coordinates of all NX trajectories.
    It can also calculate the RMSD with respect to an equilibrium geometry.

    Functions:
    ----------
       save_csv: store all loaded geometries in the raw format as a csv file.
       from_dyn (static): reads the coordinates from the NX output, dyn.out.
       from_xyz (static): reads the coordinates from the dyn.xyz file.
       read_all_trajs: store the geometries of all trajectories as np.array
       read_eq_geom: store the values of a reference geometry as a class variable
       align_geoms: aligns all geometries with respect to a reference geometry
                    using the Kabsch algorithm, and calculates the minimal RMSD.
    """

    # Defining slots to optimize performance (RAM):
    __slots__ = ['trajectories', 'labels', 'eq_xyz', 'xyz', 'rmsd', 'dataset']

    def __init__(self):
        # This variable contains a list of all available trajectories:
        # [TRAJ1, TRAJ2,..., TRAJN]
        self.trajectories = get_traj_dirs()

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

    @property
    def save_csv(self):
        if self.dataset is None:
            self.build_dataframe()

        df = self.dataset

        if self.rmsd is not None:
            df['RMSD'] = self.rmsd

        df.to_csv('all_coordinates.csv', index=False, header=True)

    def build_dataframe(self):
        if all(v is not None for v in [self.labels, self.xyz]):
            n_atoms = len(self.labels)
            col_names = [['x'+str(i), 'y'+str(i), 'z'+str(i)] for i in range(1,n_atoms+1)]
            col_names = sum(col_names, [])
            df = pd.DataFrame(self.xyz.reshape(-1, n_atoms*3), columns = col_names)

            self.dataset = df

        else:
            print("---------------------------------------")
            print("The coordinates variable is empty!")
            print("There is no data available to save.")
            print("Please run the loader functions first.")
            print("---------------------------------------")
    
    def from_dyn(self, outfile):
        read_coords = False
        t = -1
        current_time = -1
        num_atoms = np.inf

        xyz_geoms = list()
        atom_labels = list()
        counter = 0

        if self.labels is not None:
            num_atoms = len(self.labels)
            atom_labels = self.labels

        with open(outfile, 'r') as dyn_out:
            for line in dyn_out:

                # By keeping track of the current time we can deal with
                # repetitions in restart calculations, but also to deal 
                # with switching in QM/MM dynamics.
                if 'TIME' in line:
                    current_time = np.float(line.split()[-2])
                    
                if read_coords:
                    vals = line.split()
                    if len(vals) == 6:
                        counter += 1
                        if counter > num_atoms:
                            counter = 0
                            read_coords = False
                            continue
                        coords = np.array(vals[2:5], dtype=np.float64)
                        xyz_geoms.append(coords)
                        # Labels will be read only in the first iteration
                        if current_time == 0.0 and isinstance(atom_labels, list):
                            atom_labels.append(vals[0].upper())
                    else:
                        read_coords = False
                        counter = 0
                
                # This condition is used to skip repeated geometries.
                if 'geometry' in line:
                    if current_time == t:
                        read_coords = False
                    else:
                        read_coords = True
                    t = current_time

        num_atoms = len(atom_labels)

        xyz_geoms = np.array(xyz_geoms)
        xyz_geoms = xyz_geoms.reshape(-1,num_atoms,3)
        xyz_geoms *= BOHR_TO_ANG
        atom_labels = np.asarray(atom_labels)
        
        return (atom_labels, xyz_geoms)

    @staticmethod
    def from_xyz(xyzfile):
        count = 0
        xyz_geoms = list()
        atom_labels = list()
        with open(xyzfile, 'r') as inp:
            for line in inp:
                vals = line.split()
                
                if len(vals) == 1 and count == 0:
                    num_atoms = int(vals[0])
                
                if len(vals) >= 4:
                    coords = np.array(vals[-3:], dtype=np.float64)
                    coords = coords.reshape(-1,3)
                    xyz_geoms.append(coords)
                    if count <= num_atoms + 2:
                        atom_labels.append(vals[0])
                    
                count += 1
        
        atom_labels = np.asarray(atom_labels)
        xyz_array = np.vstack(xyz_geoms).reshape(-1,num_atoms,3)
        
        return (atom_labels, xyz_array)

    def read_all_trajs(self):
        all_geoms = list()
        for trj in self.trajectories:
            print("Reading geometries from %s" % trj + "...")
            if os.path.isfile(trj + '/RESULTS/dyn.out'):
                dynfile = trj + '/RESULTS/dyn.out'
                atom_labels, xyz = self.from_dyn(dynfile)
                all_geoms.append(xyz)
            elif os.path.isfile(trj + '/RESULTS/dyn.xyz'):
                xyzfile = trj + '/RESULTS/dyn.xyz'
                atom_labels, xyz = self.from_xyz(xyzfile)
                all_geoms.append(xyz)    
            else:
                print ("\nFile dyn.out or dyn.xyz not available.") 
                print("Check the directory %s" % trj + "/RESULTS" + "\n")
        
        self.xyz = np.concatenate(all_geoms, axis = 0)
        self.labels = atom_labels

    def read_eq_geom(self):
        try:
            atom_labels, ref_geom = self.from_xyz('geom.xyz')
            self.eq_xyz = np.squeeze(ref_geom, axis=0)
            self.labels = atom_labels
        except FileNotFoundError:
            print("\n-----------------------------------------------------------")
            print("Reference geometry not found!")
            print("The reference geometry is required to compute the RMSD. ")
            print("Check if the geom.xyz file is available in the current dir.")
            print("-----------------------------------------------------------\n")

    @property
    def align_geoms(self):

        if self.eq_xyz is None:
            self.read_eq_geom()

        self.eq_xyz -= rmsd.centroid(self.eq_xyz)

        if self.xyz is not None:
            xyz_dim = self.xyz.shape
            aligned_geoms = np.empty(xyz_dim, dtype=np.float64)
            rmsd_values = np.empty((xyz_dim[0]), dtype=np.float64)
        else:
            error_msg = "XYZ coordinates not loaded!" + "\n "
            error_msg += "Please make sure that the read_all_trajs function "
            error_msg += "has been executed."
            print(error_msg)
            return 
    
        for idx, geom in enumerate(self.xyz):
            geom -= rmsd.centroid(geom)
            U = rmsd.kabsch(geom, self.eq_xyz)
            geom = np.dot(geom, U)
            geom_rmsd = rmsd.rmsd(self.eq_xyz, geom)
            aligned_geoms[idx] = geom
            rmsd_values[idx] = geom_rmsd
            
        self.xyz = aligned_geoms
        self.rmsd = rmsd_values

#%% Starting new class: GetGradients
class GetGradients:

    # Defining slots to optimize performance (RAM):
    __slots__ = ['trajectories', 'all_grads', 'datasets']

    def __init__(self):
        # This variable contains a list of all available trajectories:
        # [TRAJ1, TRAJ2,..., TRAJN]
        self.trajectories = get_traj_dirs()
        # Store the gradient matrices per state as a dictionary, where
        # the key is the state identifier (e.g., {'S1': np.array})
        self.all_grads = dict()
        self.datasets = dict()

    @staticmethod
    def all_states(outfile):
        read_grads = False
        current_step = -1
        count_start = 0
        t_current = 0
        t_hop = np.inf

        grads_dict = dict()

        with open(outfile, 'r') as nx_log:
            for line in nx_log:

                # This condition is used to monitor the MD restarting, and
                # then skip the repeated gradients (Step 0 of each restart)
                if 'STARTING MOLECULAR DYNAMICS' in line:
                    count_start += 1

                if 'Nat' in line:
                    num_atoms = np.int(line.split()[-1])

                if 'STEP' in line:
                    line_list = line.replace(',','').split()
                    current_step = np.int(line_list[2])
                    t_current = np.float(line_list[4])

                if 'Time of hopping' in line:
                    t_hop = np.float(line.split()[-2])
                
                # The second condition in the if statement is used to skip
                # the gradients recalculated after hopping
                if ('Gradient' in line) and (t_current != t_hop):
                    if 'current' in line:
                       continue
                    elif 'state' in line:
                       state = 'S' + line.split()[-2]
                    else:
                       state = 'current_state'
                    
                    if count_start == 2:
                        read_grads = False
                        count_start -= 1
                    else:
                        read_grads = True
                
                    if state not in grads_dict:
                        grads_dict[state] = list()
                    continue    

                if read_grads:
                    vals = line.split()
                    if len(vals) == 3:
                        grads = np.array(vals, dtype=np.float64)
                        grads_dict[state].append(grads)
                    else:
                        read_grads = False

        for k in grads_dict.keys():
            grads_dict[k] = np.array(grads_dict[k])
            grads_dict[k] = grads_dict[k].reshape(-1,num_atoms,3)
            grads_dict[k] *= (HARTREE_TO_eV / BOHR_TO_ANG)

        return grads_dict

    def read_all_trajs(self):
        all_grads = dict()
        for trj in self.trajectories:
            print("Reading gradients from %s" % trj + "...")
            if os.path.isfile(trj + '/RESULTS/nx.log'):
                nxfile = trj + '/RESULTS/nx.log'
                gradients = self.all_states(nxfile)
                if not all_grads:
                    all_grads = {key: list() for key in gradients.keys()}
                for k in all_grads.keys():
                    all_grads[k].append(gradients[k])
            else:
                print ("\nFile nx.log is not available.") 
                print("Check the directory %s" % trj + "/RESULTS" + "\n")
        
        for k in all_grads.keys():
            self.all_grads[k] = np.concatenate(all_grads[k], axis = 0)

    def build_dataframe(self, save_csv=False):

        if self.all_grads:
            state = list(self.all_grads.keys())[0]
            n_atoms = self.all_grads[state].shape[1]
            col_names = [['Gx'+str(i), 'Gy'+str(i), 'Gz'+str(i)] for i in range(1,n_atoms+1)]
            col_names = sum(col_names, [])

            for k in self.all_grads.keys():
                df = pd.DataFrame(self.all_grads[k].reshape(-1, n_atoms*3),
                                  columns = col_names)
                self.datasets[k] = df
                if save_csv:
                    output = 'all_gradients_' + k.lower() + '.csv'
                    df.to_csv(output, index=False, header=True)

        else:
            print("---------------------------------------")
            print("The all_grads variable is empty!")
            print("There is no data available to save.")
            print("Please run the loader functions first.")
            print("---------------------------------------")        


#%% Starting new class: GetProperties
class GetProperties:
    
     __slots__ = ['trajectories', 'dataset', 'num_states']

     def __init__(self):
        self.trajectories = get_traj_dirs()
     
     def __init__(self):
         # This variable contains a list of all available trajectories:
         # [TRAJ1, TRAJ2,..., TRAJN]
         self.trajectories = get_traj_dirs()
         # This class variable will be used to store a dataframe
         # with all properties read from the NX outputs
         self.dataset = None
         # Auxiliary variable to keep track of the number of states.
         # The default value will be updated in the energy function.
         self.num_states = None

     @property
     def save_csv(self):
        if self.dataset is not None:
            df = self.dataset.copy()
            df['time'] = df['time'].astype(object)
            df.to_csv('all_properties.csv', index=False, header=True, 
                      float_format="%.10f")
            
        else:
            print("The properties variable is empty!")
            print("There is no data to save.")
            print("Please run the loader functions first.")

     def _update_properties(self,df):

         if self.dataset is not None:
             if self.dataset.shape[0] == df.shape[0]:
                 if 'time' in self.dataset.columns.tolist():
                     dfs_to_merge = (self.dataset,df)
                 else:
                     dfs_to_merge = (df,self.dataset)    
                 self.dataset = pd.concat(dfs_to_merge, axis=1)
             else:
                 warning = "***************************************************************\n"
                 warning += "WARNING: The size of the dataframes does not match!\n"
                 warning += "         Please check if there are repeated or missing lines\n"
                 warning += "         in one of the data files.\n"
                 warning += "***************************************************************"
                 print(warning) 
         else:
             print("The properties dataset is empty.") 
             print("Updating class variable with the current loaded data.")
             self.dataset = df

     def energies(self):

         all_energies = list()
         traj_id = list()
         
         for trj in self.trajectories:
             print('Reading energies from %s' % trj)
             try:
                 enfile = trj + '/RESULTS/en.dat'
                 en = np.loadtxt(enfile)
             except:
                 print('\n-------------------------------------')
                 print("Energy file not found or corrupted.")
                 print("Check the %s/RESULTS directory." % trj)
                 print('-------------------------------------\n')
                 continue

             all_energies.append(en)
             n_samples = en.shape[0]
             idx = np.int(trj.replace('TRAJ',''))
             traj_id.append(np.full(n_samples, idx, dtype=np.int))   

         traj_id = np.hstack(traj_id)
         all_energies = np.vstack(all_energies)
         all_energies[:,1:] *= HARTREE_TO_eV

         self.num_states = all_energies.shape[1] - 3
         state_labels = ['S' + str(i+1) for i in range(self.num_states)]
         col_names = ['time'] + state_labels + ['Current_State', 'Total_Energy']
         df = pd.DataFrame(all_energies, columns=col_names)

         df['TRAJ'] = traj_id

         df['State'] = df[state_labels].eq(df['Current_State'], axis=0).idxmax(1)
         df['State'] = df['State'].str.replace('S','').astype(int)
         # Auxiliary variable used to find the hopping points
         df['State_Next'] = df.groupby(by='TRAJ')['State'].shift(-1, fill_value=-10)
         
         # Loop to calculate energy difference between all possible pair of states
         for i,j in combinations(state_labels, 2):
             from_to = j.replace('S', '') + i.replace('S', '')
             new_col = 'DE' + from_to
             df[new_col] = df[j] - df[i]
             si = np.int(i.replace('S', ''))
             sj = np.int(j.replace('S', ''))
             # Create a binary column to identify hopping geometries 
             # The first condition corresponds to hoppings by state decay
             new_col = 'Hops_S' + str(sj) + str(si)
             condition = (df['State'] == sj) & (df['State_Next'] == si)
             df[new_col] = np.where(condition, 1, 0)
             # The second condition takes into account upward hoppings 
             new_col = 'Hops_S' + str(si) + str(sj)
             condition = (df['State'] == si) & (df['State_Next'] == sj)
             df[new_col] = np.where(condition, 1, 0)

         cols_to_drop = state_labels[1:] + ['Current_State', 'State_Next']
         df.drop(cols_to_drop, axis = 1, inplace = True)

         # Remove all columns that contains only zeros
         df = df.loc[:, (df != 0).any(axis=0)]

         self._update_properties(df)
         df = self.dataset

         return df

     def oscillator_strength(self):

         for trj in self.trajectories:
             step = -1
             counter = -1
             read_line = False
             
             print('Reading properties from %s' % trj)   
             try:
                 propfile = trj + '/RESULTS/properties'
                 f = open(propfile, 'r')
             except:
                 print('\n---------------------------------------')
                 print("Properties file not found or corrupted.")
                 print("Check the %s/RESULTS directory." % trj)
                 print('---------------------------------------\n')
                 continue

             lines = f.read()
             if "Oscillator strength" not in lines:
                 f.close()
                 print("\nOscillator strength not found in %s" % propfile)
                 return
             else:    
                 lines = lines.split('\n')
                 
                 oscillator_lines = list(filter(lambda x: x.startswith(' Oscillator '), lines))
                 time_lines = list(filter(lambda x: x.startswith(' TIME '), lines))
                 time_lines = list(set([float(i.split()[2]) for i in time_lines]))
                 num_rows = len(time_lines)
                 osc_dict = dict()
                 for line in oscillator_lines:
                     states = line.split()[2]
                     if states not in osc_dict.keys():
                         osc_dict[states] = np.full(num_rows, np.nan)

                 # Start reading the properties file
                 for line in lines:
                    if 'STEP:' in line:
                        current_step = np.int(line.split()[4])
                        if current_step != step:
                            read_line = True
                            counter += 1
                        else:
                            read_line = False
                        step = current_step

                    if ('Oscillator' in line) and read_line:
                        x = np.float(line.split()[4])
                        states = line.split()[2]
                        osc_dict[states][counter] = x

             f.close()

         col_names = ['f_' + ''.join(key.replace('(','').replace(')','').split(',')) 
                      for key in osc_dict]
         df = pd.DataFrame(osc_dict)
         df.columns = col_names
         
         self._update_properties(df)
         df = self.dataset

         return df

     def populations(self): 
         
         coefs_list = list()
         
         for trj in self.trajectories:
             
             print('Reading populations from %s' % trj)   
             try:
                 dynfile = trj + '/RESULTS/dyn.out'
                 f = open(dynfile, 'r')
             except:
                 print('\n-------------------------------------------------')
                 print("The file dyn.out was not found or is corrupted.")
                 print("Check the %s/RESULTS directory." % trj)
                 print('-------------------------------------------------\n')
                 continue

             n_states = 0
             lines = f.readlines()
             # Start reading the properties file
             for line in lines:
                 if 'STEP' in line:
                     current_step = np.int(line.split()[1])
                 if ' Wave function state ' in line:
                        coefs = list(np.float_(line.split()[-2:]))
                        coefs_list.append(coefs)
                        if current_step == 0:
                            n_states += 1

         coefs_list = np.array(coefs_list, dtype = np.float64)
         pop = np.sum(coefs_list**2, axis = 1).reshape(-1,n_states)

         ncols = pop.shape[1]
         col_names = ['Pop' + str(i) for i in range(1,ncols+1)]
         df = pd.DataFrame(pop, columns=col_names)

         self._update_properties(df)
         df = self.dataset

         return df
