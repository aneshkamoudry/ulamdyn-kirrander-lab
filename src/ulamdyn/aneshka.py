"""Module with auxiliary functions and constants used to process dataframe to .xyz data and visualisaiton."""

# Author: Aneshka Moudry <aneshkamoudry@gmail.com>
# Date: March 4th 2025

import numpy as np
import rmsd
try:
    import modin.pandas as pd

except ModuleNotFoundError:
    import pandas as pd

from ulamdyn.data_loader import GetCoords, GetProperties
from ulamdyn.base import BaseClass
from pathlib import Path
from ulamdyn.descriptors import ZMatrix

__all__ = ['DataFrameToArray', 'ConicalIntersectionClassifier']

# %% Starting the first class - visualise trajectories from dataframe

class DataFrameToArray(GetCoords):
    """
    A class that converts any pandas DataFrame containing 'TRAJ' and 'time' to a geom.xyz format NumPy array with additional
    data processing capabilities.
    """

    # __slots__ = 

    def __init__(self, dataframe, all_geoms=None) -> None:
        """
        Initialize the class with a pandas DataFrame.
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The input DataFrame to be converted
        """

        super().__init__()
        self.dataframe = dataframe
        if all_geoms is None:
            self.read_all_trajs()
        elif isinstance(all_geoms, GetCoords):
            self.xyz = all_geoms.xyz
            self.traj_time = all_geoms.traj_time
        elif isinstance(all_geoms, np.ndarray):
            self.xyz = all_geoms

    def convert_df_tuple(self):
        """
        Convert the DataFrame to a list of tuples containing the trajectory indeces and timesteps.
        
        Returns:
        --------
        list 
            A list of tuples [int, float] containing the trajectory index and the time step.
        """
        return [tuple(row) for row in self.dataframe[['TRAJ', 'time']].values]
    
    def convert_df_geomstringlist(self):
        """
        Convert a df to a list of geometry strings.
        
        Returns:
        --------
        list
            A list of geometry strings.
        """
        tuple_list = self.convert_df_tuple()
        geometry_strings = []

        for idx in tuple_list:
            geometry_string = self[idx]
            geometry_strings.append(geometry_string)

        return geometry_strings
    
    
# %% Conical intersection classifier

class ConicalIntersectionClassifier(BaseClass):

    __slots__ = ["dataset", "ci_refs"]

    def __init__(self, dataframe=None) -> None:
        """Class initializer."""
        self.properties = GetProperties()
        self.coords = GetCoords()
        
        if dataframe is None:
            self.properties.energies()
        elif isinstance(dataframe, GetProperties):
            self.properties.dataset = dataframe
        self.ci_refs = None

    def check_xyz_files(self) -> bool:
        """
        Check if all files in a directory are XYZ files.
        This function not only checks file extensions but also verifies if each file
        follows basic XYZ format structure.
    
        Parameters:
            directory (str): Path to the directory to check
    
        Returns:
            bool: True if all files are valid XYZ files, False otherwise
        """
        dir_path = Path('ci_refs')
        files = [f for f in dir_path.iterdir() if f.is_file() and not f.name.startswith('.')]
    
        if not files:
            print(f"No files found in ci_refs directory.")
            return False
    
        for file_path in files:
            # First check file extension
            if file_path.suffix.lower() != '.xyz':
                print(f"Found non-XYZ file: {file_path.name}")
                return False
            
        try:
            with open(file_path, 'r') as f:
                # Read first line which should be number of atoms
                first_line = f.readline().strip()
                if not first_line.isdigit():
                    print(f"Invalid XYZ format in {file_path.name}: First line should be number of atoms")
                    return False
        except Exception as e:
            print(f"Error reading file {file_path.name}: {str(e)}")
            return False
            
        return True
    
    def load_ci_refs(self) -> dict:
        """
        Load the reference files for the conical intersection classification.
         Returns:
        dict: Dictionary mapping CI IDs (file stems) to their xyz coordinates
        """
        ci_refs = {}

        if self.check_xyz_files():
            dir_path = Path('ci_refs')
            files = [f for f in dir_path.iterdir() if f.is_file() and not f.name.startswith('.')]

            for file in files:
                ci_id = file.stem
                _, xyz = self.coords.from_xyz(file)
                ci_refs[ci_id] = xyz
                self.ci_refs = ci_refs
                     
        else:
            print("No reference .xyz files found.")
            return
    
    def convert_df_tuple(self):
        """
        Convert the DataFrame to a list of tuples containing the trajectory indeces and timesteps.
        
        Returns:
        --------
        list 
            A list of tuples [int, float] containing the trajectory index and the time step.
        """
        return [tuple(row) for row in self.properties.dataset[['TRAJ', 'time']].values]
    
    def convert_df_geomstringlist(self):
        """
        Convert a df to a list of geometry strings.
        
        Returns:
        --------
        list
            A list of geometry strings.
        """
        tuple_list = self.convert_df_tuple()
        geometry_strings = []

        for idx in tuple_list:
            geometry_string = self.coords[idx]
            geometry_strings.append(geometry_string)

        return geometry_strings
    
    @staticmethod
    def calculate_angle(
        P: np.array, 
        A: int,
        B: int,
        C: int
        ) -> float:
        """
        Calculate the angle between three atoms where B is the center atom. 
        A, B, C are integers corresponding to order of desired atoms in XYZ file.

        Parameters:
        -----------
         P : array
           (N,D) matrix, where N is points and D is dimension.
         A : integer
        B : integer  
        C : integer
        
        Returns:
        --------
        float
            Angle in degrees
        """

        vector1 = P[A] - P[B]
        vector2 = P[C] - P[B]
    
        vector1_norm = vector1 / np.linalg.norm(vector1)
        vector2_norm = vector2 / np.linalg.norm(vector2)
    
        dot_product = np.dot(vector1_norm, vector2_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)
        angle = float(np.degrees(angle))
    
        return angle
    
    @staticmethod
    def calculate_angle_vectors( 
        A: np.array,
        B: np.array,
        C: np.array,
        ) -> float:
        """
        Calculate the angle between three atoms where B is the center atom. 
        A, B, C are arrays corresponding to the XYZ coordinates of the atom.

        Parameters:
        -----------

        A : array
           (1,3) matrix
        B : array
           (1,3) matrix  
        C : array
           (1,3) matrix
        
        Returns:
        --------
        float
            Angle in degrees
        """

        vector1 = A - B
        vector2 = C - B
    
        vector1_norm = vector1 / np.linalg.norm(vector1)
        vector2_norm = vector2 / np.linalg.norm(vector2)
    
        dot_product = np.dot(vector1_norm, vector2_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)
        angle = float(np.degrees(angle))
    
        return angle
    
    def classify_ci_geom(self, geom_data, method='rmsd', remove_hydrogens=False, remove_atoms=None) -> str:
        """
        Classify a single conical intersection geometry.
    
        Parameters:
        -----------
        geom_data : tuple or np.ndarray
            Either a tuple of (coordinates, atom_types) or just coordinates
        method : str, default='rmsd'
            Method to use for classification ('rmsd' or 'angle')
        remove_hydrogens : bool, default=False
            Whether to remove hydrogen atoms before RMSD calculation
        remove_atoms : list or None, default=None
            List of atom indices to remove before RMSD calculation
        
        Returns:
        --------
        str
            Classification key
        """
        if self.ci_refs is None:
            self.load_ci_refs()

        ci_refs = self.ci_refs

        if isinstance(geom_data, tuple) and len(geom_data) == 2:
                geom = geom_data[0]
                atom_types = geom_data[1]
        else:
            geom = geom_data
            atom_types = None

        if method == 'rmsd':
            
            if remove_hydrogens or remove_atoms is not None:
                mask = np.ones(geom.shape[0], dtype=bool)

                if remove_hydrogens and atom_types is not None:
                    h_indices = [i for i, atom in enumerate(atom_types) if atom.lower() == 'h']
                    mask[h_indices] = False

                if remove_atoms is not None:
                    mask[remove_atoms] = False
                
                filtered_geom = geom[mask]
            else:
                filtered_geom = geom
            
            filtered_geom -= rmsd.centroid(filtered_geom)

            rmsd_dict = {}

            for key in ci_refs:
                ref_geom = ci_refs[key]
                ref_geom = ref_geom.squeeze()
                ref_geom -= rmsd.centroid(ref_geom)
                u = rmsd.kabsch(ref_geom, geom)
                aligned_geom = np.dot(geom, u)
                rmsd_dict[key] = rmsd.rmsd(ref_geom, aligned_geom)
            min_key = min(rmsd_dict, key=rmsd_dict.get)
            
        elif method == 'angle':
            geom = geom.squeeze()
            bangle = {
                        'a': (1, 3, 4),
                        'b': (3, 4, 2),
                        'c': (2, 1, 3),
                        'd': (4, 2, 1)
                    }
            
            for k, v in bangle.items():
                bangle[k] = self.calculate_angle(geom, (v[0]-1), (v[1]-1), (v[2]-1))

            min_key = min(bangle, key=bangle.get)
    
        return min_key    
    
    def get_xyz_df(self, idx: tuple) -> np.ndarray:
        """
        Get XYZ coordinates as a numpy array.
    
        Parameters
        ----------
        idx : tuple
           Tuple of (trajectory, time)
    
        Returns
        -------
        np.ndarray
            Array of shape (n_atoms, 3) containing XYZ coordinates
        """
        if self.coords.xyz is None:
            self.coords.read_all_trajs()
        
        traj, time = idx
        c1 = self.coords.traj_time[:, 0] == traj
        c2 = self.coords.traj_time[:, 1] == time
        selected_geom = self.coords.xyz[(c1 & c2)][0]
        label = self.coords.labels
        return selected_geom, label

    def classify_ci_df(self, hop: str, method='rmsd', remove_hydrogens=False, remove_atoms=None) -> pd.DataFrame:
        """
        Classify the conical intersection geometries.
    
        Parameters:
        -----------
        hop : str
            Column name for hopping data in the dataset
        method : str, default='rmsd'
            Method to use for classification ('rmsd' or 'angle')
        remove_hydrogens : bool, default=False
            Whether to remove hydrogen atoms before RMSD calculation
        remove_atoms : list or None, default=None
            List of atom indices to remove before RMSD calculation
        
        Returns:
        --------
        pandas.DataFrame: Dataframe containing the classified geometries
        """
        if self.ci_refs is None:
            self.load_ci_refs()
        
        if self.properties.dataset is None:
            self.properties.energies()
        
        df = self.properties.dataset
        
        if hop not in df.columns:
            print("No hopping data found in the dataset.")
            return
        
        else: 
            ci_classified = []
            for idx, row in df.iterrows():
                if row[hop] == 1:
                    geom_data = self.get_xyz_df([row['TRAJ'], row['time']])

                    if method == 'rmsd':
                        ci_id = self.classify_ci_geom(geom_data, 'rmsd',
                                                    remove_hydrogens=remove_hydrogens,
                                                    remove_atoms=remove_atoms)

                        ci_classified.append(ci_id)

                    elif method == 'angle':
                        ci_id = self.classify_ci_geom(geom_data, 'angle')
                        ci_classified.append(ci_id)
                        
                else:
                    ci_classified.append('None')
                    
            df['CI_ID'] = ci_classified
   
        return df
    
    def internal_coordinates_nbd_qc(self) -> pd.DataFrame:
        """
        Calculate the internal coordinates specific to the NBD/QC molecular system
        
        Returns:
        --------
        pandas.DataFrame
            Dataframe containing the internal coordinates.
        """
        if self.properties.dataset is None:
            self.properties.energies()
        
        df = self.properties.dataset

        # Calculate the internal coordinates - 'square' angles

        square_angle = {
                        'a134': (1, 3, 4),
                        'a342': (3, 4, 2),
                        'a213': (2, 1, 3),
                        'a421': (4, 2, 1)
                }
        
        for k, v in square_angle.items():
            angles = []

            for idx, row in df.iterrows():
                geom_data = self.get_xyz_df([row['TRAJ'], row['time']])
                geom = geom_data[0]
                angle = self.calculate_angle(geom, (v[0]-1), (v[1]-1), (v[2]-1))
                angles.append(angle)
            
            df[f"a_{k}"] = angles

        # 'Tent' angle

        tent_angle = []
        for idx, row in df.iterrows():
            geom_data = self.get_xyz_df([row['TRAJ'], row['time']])
            geom = geom_data[0]
            midpoint13 = (geom[0] + geom[2])/2
            c7 = geom[6]
            midpoint24 = (geom[1] + geom[3])/2
            angle = self.calculate_angle_vectors(midpoint13, c7, midpoint24)
            tent_angle.append(angle)

        df['a_tent'] = tent_angle

        # 'Bridge' angle

        bridge_angle = []

        for idx, row in df.iterrows():
            geom_data = self.get_xyz_df([row['TRAJ'], row['time']])
            geom = geom_data[0]
            angle = self.calculate_angle(geom, 4, 6, 5)
            bridge_angle.append(angle)
        
        df['$\theta$'] = bridge_angle

        # 'triangular' angle

        triangular_angle = {
                        'a512': (5, 1, 2),
                        'a521': (5, 2, 1),
                        'a634': (6, 3, 4),
                        'a643': (6, 4, 3)
                    }

        for k, v in square_angle.items():
            angles = []

            for idx, row in df.iterrows():
                geom_data = self.get_xyz_df([row['TRAJ'], row['time']])
                geom = geom_data[0]
                angle = self.calculate_angle(geom, (v[0]-1), (v[1]-1), (v[2]-1))
                angles.append(angle)
            
            df[f"a_{k}"] = angles

        # 'book' angle

        book_angle = []
        for idx, row in df.iterrows():
            geom_data = self.get_xyz_df([row['TRAJ'], row['time']])
            geom = geom_data[0]
            
            vector1 = geom[4] - geom[0]
            vector2 = geom[2] - geom[0]
            plane_vector_1 = np.cross(vector1, vector2)
            vector3 = geom[4] - geom[1]
            vector4 = geom[3] - geom[1]
            plane_vector_2 = np.cross(vector3, vector4)

            plane_vector1_norm = plane_vector_1 / np.linalg.norm(plane_vector_1)
            plane_vector2_norm = plane_vector_2 / np.linalg.norm(plane_vector_2)
    
            dot_product1 = np.dot(plane_vector1_norm, plane_vector2_norm)
            dot_product1 = np.clip(dot_product1, -1.0, 1.0)
            angle1 = np.arccos(dot_product1)
            angle1 = float(np.degrees(angle1))
            
            vector5 = geom[5] - geom[2]
            vector6 = geom[0] - geom[2]
            plane_vector_3 = np.cross(vector5, vector6)
            vector7 = geom[5] - geom[3]
            vector8 = geom[1] - geom[3]
            plane_vector_4 = np.cross(vector7, vector8)

            plane_vector3_norm = plane_vector_3 / np.linalg.norm(plane_vector_3)
            plane_vector4_norm = plane_vector_4 / np.linalg.norm(plane_vector_4)
    
            dot_product2 = np.dot(plane_vector3_norm, plane_vector4_norm)
            dot_product2 = np.clip(dot_product2, -1.0, 1.0)
            angle2 = np.arccos(dot_product2)
            angle2 = float(np.degrees(angle2))

            angle = (angle1 + angle2) / 2
            book_angle.append(angle)

        df['a_book'] = book_angle

        # r_base length

        r_base = []
        for idx, row in df.iterrows():
            geom_data = self.get_xyz_df([row['TRAJ'], row['time']])
            geom = geom_data[0]
            vec1 = geom[1] - geom[0]
            vec2 = geom[4] - geom[1]
            r = vec1 + vec2
            r_dot = np.dot(r, vec1) * vec1 / np.linalg.norm(vec1)
            r_bases = np.linalg.norm(r - r_dot)
            vec3 = geom[3] - geom[2]
            vec4 = geom[5] - geom[3]
            r1 = vec3 + vec4
            r_dot1 = np.dot(r1, vec3) * vec3 / np.linalg.norm(vec3)
            r_bases1 = np.linalg.norm(r1 - r_dot1)

            r_ave = (r_bases + r_bases1) / 2

            r_base.append(r_ave)

        df['r_base'] = r_base

        # dihedral angles

        dihedral_angle = {
                        'd6421': (6, 4, 2, 1),
                        'd6312': (6, 3, 1, 2),
                        'd5134': (5, 1, 3, 4),
                        'd5243': (5, 2, 4, 3)
                    }

        for k, v in dihedral_angle.items():
            angles = []

            for idx, row in df.iterrows():
                geom_data = self.get_xyz_df([row['TRAJ'], row['time']])
                geom = geom_data[0]
                angle = ZMatrix.get_dihedral(geom, [v[0]-1, v[1]-1, v[2]-1, v[3]-1])
                angles.append(angle)
            
            df[f"d_{k}"] = angles

        df.to_csv('out.csv.gz', compression='gzip')

        return df

        