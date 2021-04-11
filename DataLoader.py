## Author: Max Pinheiro Jr <maxjr82@gmail.com>
## Date: 03/10/2021

import os
import sys
import glob
import rmsd
import shutil
import numpy as np
import pandas as pd
import pandas as pd

from itertools import combinations

filedir = os.path.dirname(__file__)

BOHR_TO_ANG = 0.529177210903
HARTREE_TO_KCAL = 627.5096080305927
HARTREE_TO_eV = 27.211399


# This function is used to return one list with all 'TRAJXX' directories
# sorted in ascending order.
def get_traj_dirs():
    dirs_list = glob.glob("TRAJ*")
    dirs_list = sorted(dirs_list, key = lambda x: int(x.rsplit("TRAJ")[1]))
    return dirs_list

class GetCoords:
    """
    This class object read the Cartesian coordinates for all NX trajectories,
    and calculate the RMSD with respect to an equilibrium geometry.

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
    __slots__ = ['trajectories', 'labels', 'eq_xyz', 'xyz', 'rmsd']

    def __init__(self):
        self.trajectories = get_traj_dirs()
        self.labels = None
        self.eq_xyz = None
        self.xyz = None
        self.rmsd = None

    def build_dataframe(self, save_csv=False):
        if all(v is not None for v in [self.labels, self.xyz]):
            n_atoms = len(self.labels)
            col_names = [['x'+str(i), 'y'+str(i), 'z'+str(i)] for i in range(1,n_atoms+1)]
            col_names = sum(col_names, [])
            df = pd.DataFrame(self.xyz.reshape(-1, n_atoms*3), columns = col_names)
            
            if self.rmsd is not None:
                df['RMSD'] = self.rmsd

            if save_csv:
                df.to_csv('all_coordinates.csv', index=False, header=True)

        else:
            print("The coordinates variable is empty!")
            print("There is no data to save.")
            print("Please run the loader functions first.")
            sys.exit() 

        return df    
    
    @staticmethod
    def from_dyn(outfile):
        read_coords = False
        step = -1
        current_step = -1

        xyz_geoms = list()
        atom_labels = list()

        with open(outfile, 'r') as dyn_out:
            for line in dyn_out:

                if 'STEP' in line:
                    current_step = np.int(line.split()[1])
                    
                if read_coords:
                    vals = line.split()
                    if len(vals) == 6:
                        coords = np.array(vals[2:5], dtype=np.float64)
                        xyz_geoms.append(coords)
                        # Labels will be read only in the first iteration
                        if current_step == 0:
                            atom_labels.append(vals[0].upper())
                    else:
                        read_coords = False
                        
                if 'geometry' in line:
                    if current_step < step:
                        read_coords = False
                    else:
                        read_coords = True
                    step = current_step

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
            print("Reading geometries from %s" % trj + "...\n")
            if os.path.isfile(trj + '/RESULTS/dyn.out'):
                dynfile = trj + '/RESULTS/dyn.out'
                atom_labels, xyz = self.from_dyn(dynfile)
                all_geoms.append(xyz)
            elif os.path.isfile(trj + '/RESULTS/dyn.xyz'):
                xyzfile = trj + '/RESULTS/dyn.xyz'
                atom_labels, xyz = self.from_xyz(xyzfile)
                all_geoms.append(xyz)    
            else:
                print ("File dyn.xyz or dyn.out not found!") 
                print("Please check the directory %s" % trj + "/RESULTS" + "\n")
        
        self.xyz = np.concatenate(all_geoms, axis = 0)
        self.labels = atom_labels

    def read_eq_geom(self):
        try:
            _, ref_geom = self.from_xyz('geom.xyz')
        except FileNotFoundError:
            print("A reference geometry file must be provided!")
            print("Check if the geom.xyz file is available in the current dir.")

        self.eq_xyz = np.squeeze(ref_geom, axis=0)

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
            return error_msg    
    
        for idx, geom in enumerate(self.xyz):
            geom -= rmsd.centroid(geom)
            U = rmsd.kabsch(geom, self.eq_xyz)
            geom = np.dot(geom, U)
            geom_rmsd = rmsd.rmsd(self.eq_xyz, geom)
            aligned_geoms[idx] = geom
            rmsd_values[idx] = geom_rmsd
            
        self.xyz = aligned_geoms
        self.rmsd = rmsd_values

class GetProperties:
    
     __slots__ = ['trajectories', 'properties', 'num_states']
     
     def __init__(self):
         self.trajectories = get_traj_dirs()
         # This class variable will be used to store a dataframe
         # with all properties read from the NX outputs
         self.properties = None
         # Auxiliary variable to keep track of the number of states.
         # The default value will be updated in the energy function.
         self.num_states = None

     @property
     def save_csv(self):
        if self.properties is not None:
            df = self.properties.copy()
            df['time'] = df['time'].astype(object)
            df.to_csv('all_properties.csv', index=False, header=True, float_format="%.8f")
            
        else:
            print("The properties variable is empty!")
            print("There is no data to save.")
            print("Please run the loader functions first.")
            sys.exit()

     def _update_properties(self,df):

         if self.properties is not None:
             if self.properties.shape[0] == df.shape[0]:
                 if 'time' in self.properties.columns.tolist():
                     dfs_to_merge = (self.properties,df)
                 else:
                     dfs_to_merge = (df,self.properties)    
                 self.properties = pd.concat(dfs_to_merge, axis=1)
             else:
                 warning = "****************************************************************\n"
                 warning += "WARNING: The size of the dataframes does not match!\n"
                 warning += "         Please check if there are repeated or missing lines"
                 warning += "         in one of the data files."
                 warning += "\n****************************************************************"
                 print(warning) 
         else:
             print("Properties is empty.") 
             print("Its value will be updated with the current loaded data.")
             self.properties = df

     def energies(self) -> pd.core.frame.DataFrame:

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
         df = self.properties

         return df

     def oscillator_strength(self) -> pd.core.frame.DataFrame:
         
         osc = list()
         n_samples = 0
         
         for trj in self.trajectories:
             step = -1
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
                 print("Oscillator strength not found in %s" % propfile)
                 return
             else:    
                 lines = lines.split('\n')
                 # Start reading the properties file
                 for line in lines:
                    if 'STEP:' in line:
                        current_step = np.int(line.split()[4])
                        if current_step != step:
                            read_line = True
                            n_samples += 1
                        else:
                            read_line = False
                        step = current_step

                    if ('Oscillator' in line) and read_line:
                        x = np.float(line.split()[4])
                        osc.append(x)
             f.close()

         osc = np.array(osc, dtype=np.float64)
         osc = osc.reshape(n_samples,-1)

         ncols = osc.shape[1]
         col_names = ['f1' + str(i) for i in range(2,ncols+2)]
         df = pd.DataFrame(osc, columns=col_names)

         self._update_properties(df)
         df = self.properties

         return df

     def populations(self) -> pd.core.frame.DataFrame: 
         
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
         df = self.properties

         return df
