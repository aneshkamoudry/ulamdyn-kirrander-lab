## Author: Max Pinheiro Jr <maxjr82@gmail.com>
## Date: 03/10/2021

import os
import sys
import numpy as np
import pandas as pd

from DataLoader import GetCoords
from itertools import combinations

filedir = os.path.dirname(__file__)

class R2(GetCoords):
    """
    Construct the R2 descriptor defined as the Euclidean distances between
    all non-equivalent pair of atoms.

    Functions:
    ----------
        xyz_to_distances: calculates the R2 distances vector for each XYZ matrix
        build_descriptor: return a dataframe with the R2 descriptors for all molecules
    """
    def __init__(self):
        self.r2_ref_geom = None

    def xyz_to_distances(self,xyz_matrix: np.ndarray) -> np.ndarray:

        n_atoms = len(xyz_matrix)
        distance_matrix = np.zeros((n_atoms,n_atoms))

        for i,j in combinations(range(len(xyz_matrix)),2):
            R = np.linalg.norm(xyz_matrix[i]-xyz_matrix[j])
            distance_matrix[j,i] = R
         
        r2_vector = distance_matrix[np.tril_indices(len(distance_matrix),-1)]

        return r2_vector

    def build_descriptor(self, all_geoms: np.ndarray, delta = False,
                         save_csv=False) -> pd.core.frame.DataFrame:

        n_samples, n_atoms, _ = all_geoms.shape
        id_atom_pairs = np.tril_indices(n_atoms,-1)
        n_features = len(id_atom_pairs[0])

        r2_descriptor = np.empty((n_samples, n_features), dtype=np.float64)

        for i, xyz in enumerate(all_geoms):
            d = self.xyz_to_distances(xyz)
            r2_descriptor[i] = d

        col_names = list(map(lambda x,y: 'r' + str(y+1) + str(x+1),\
             id_atom_pairs[0], id_atom_pairs[1]))

        self.read_eq_geom()
        eq_geom = self.eq_xyz.copy()
        self.r2_ref_geom = self.xyz_to_distances(eq_geom)

        if delta:
            r2_descriptor = r2_descriptor - self.r2_ref_geom

        df_r2 = pd.DataFrame(r2_descriptor, columns=col_names)

        if save_csv:
            df_r2.to_csv("pairwise_distances.csv", index=False)

        return df_r2

class ZMatrix(GetCoords):
    """
    Construct a molecular descriptor derived from the Z-Matrix.

    Functions:
    ----------
        get_distance: calculates the distances between two atoms
        get_angle: calculate the angle bewtween three atoms
        get_dihedral: calculate the dihedral bewtween four atoms
        build_descriptor: return a dataframe with the Z-matrix for all molecules
    """

    # Defining slots to optimize performance (RAM):
    __slots__ = ['distancematrix', 'connectivity', 'angleconnectivity', 
                 'dihedralconnectivity', 'zmat_ref_geom']

    def __init__(self):

        self.distancematrix = None

        # Internal Coordinate Connectivity
        self.connectivity = None
        self.angleconnectivity = None
        self.dihedralconnectivity = None

        self.zmat_ref_geom = None

    @staticmethod
    def get_distance(geom: np.ndarray, idx_atoms) -> np.float:
        """
        Auxiliary function to calculate the Euclidean distance between 
        pair of atoms.
        """
        i, j = idx_atoms
        vec = geom[j] - geom[i]
        dist = np.linalg.norm(vec)

        return dist

    @staticmethod
    def get_angle(geom: np.ndarray, idx_atoms: list) -> np.float:
        """
        Auxiliary function to calculate the angle between three atoms 
        The output angle is given in degrees.
        """
        i, j, k = idx_atoms
        rij = geom[i] - geom[j]
        rkj = geom[k] - geom[j]
        cos_theta = np.dot(rij, rkj)
        sin_theta = np.linalg.norm(np.cross(rij, rkj))
        theta = np.arctan2(sin_theta, cos_theta)
        theta = np.degrees(theta)

        return theta

    @staticmethod
    def get_dihedral(geom: np.ndarray, idx_atoms: list) -> np.float:
        """
        This function calculates the dihedral angle between four atoms 
        using the praxeolitic formula: 1 sqrt, 1 cross product.
        The output angle is given in degrees.
        """

        if not isinstance(idx_atoms, list):
            idx_atoms = list(idx_atoms)

        (p0, p1, p2, p3) = geom[idx_atoms]
        
        b0 = -1.0*(p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        # normalize b1 so that it does not influence magnitude of vector
        # rejections that come next
        b1 /= np.linalg.norm(b1)

        # vector rejections
        # v = projection of b0 onto plane perpendicular to b1
        #   = b0 minus component that aligns with b1
        # w = projection of b2 onto plane perpendicular to b1
        #   = b2 minus component that aligns with b1
        v = b0 - np.dot(b0, b1)*b1
        w = b2 - np.dot(b2, b1)*b1

        # angle between v and w in a plane is the torsion angle
        # v and w may not be normalized but that's fine since tan is y/x
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)

        phi = np.degrees(np.arctan2(y, x))

        return phi

    @staticmethod
    def get_bending(geom: np.ndarray, idx_atoms: list):
        """
        This function calculates the bending angle between two different
        planes of the molecule defined by two sets of three atoms.
        The output angle is given in degrees.
        """

        idx_ring1, idx_ring2 = idx_atoms
        
        # Calculate the vector perpendicular to the plan of the first ring
        v1_ring1 = (geom[idx_ring1[2]] - geom[idx_ring1[0]])/2
        v2_ring1 = (geom[idx_ring1[1]] - geom[idx_ring1[0]])/2
        normal_to_ring1 = np.cross(v2_ring1,v1_ring1)
        normal_to_ring1 /= np.linalg.norm(normal_to_ring1)

        # Calculate the vector perpendicular to the plan of the second ring
        v1_ring2 = (geom[idx_ring2[2]] - geom[idx_ring2[0]])/2
        v2_ring2 = (geom[idx_ring2[1]] - geom[idx_ring2[0]])/2
        normal_to_ring2 = np.cross(v2_ring2,v1_ring2)
        normal_to_ring2 /= np.linalg.norm(normal_to_ring2)

        # Calculate the angle between the two normal vectors
        cos_theta = normal_to_ring1.dot(normal_to_ring2)
        ang_between_rings = np.arccos(cos_theta)
        ang_between_rings = np.degrees(ang_between_rings)
        
        return ang_between_rings    

    def _build_distance_matrix(self, xyz_matrix: np.ndarray):
        
        n_atoms = xyz_matrix.shape[0]
        self.distancematrix = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in [x for x in range(n_atoms) if x > i]:
                self.distancematrix[i][j] = np.linalg.norm(xyz_matrix[i] - xyz_matrix[j])
                self.distancematrix[j][i] = self.distancematrix[i][j]     

    def build_descriptor(self, all_geoms: np.ndarray, delta = False, 
                         save_csv=False) -> pd.core.frame.DataFrame:
        """
       'Z-Matrix Algorithm'
        Build main components of zmatrix:
        Connectivity vector
        Distances between connected atoms (atom >= 1)
        Angles between connected atoms (atom >= 2)
        Dihedral angles between connected atoms (atom >= 3)
        """

        # Use a function inherited from GetCoords to read the coordinates
        # of a reference geometry
        self.read_eq_geom()
        eq_geom = self.eq_xyz.copy()
        n_atoms = eq_geom.shape[0]

        # Compute the R2 distance matrix for the reference geometry
        self._build_distance_matrix(eq_geom)

        # The connectivity variables store tuples with the indices of the 
        # connected atoms based on a distance criterion of proximity
        self.connectivity = [(0,0) for _ in range(0,n_atoms)]
        self.angleconnectivity = [(0,0,0) for _ in range(0,n_atoms)]
        self.dihedralconnectivity = [(0,0,0,0) for _ in range(0,n_atoms)]

        distances = list()
        angles = list()
        dihedrals = list()

        # This first loop goes over the atoms of the reference geometry 
        # to obtain the Z-Matrix and all indices of "connected" atoms
        for atom in range(1,n_atoms):
            # For current atom, find the nearest atom among previous atoms
            distvector = self.distancematrix[atom][:atom]
            distmin = np.array(distvector[np.nonzero(distvector)]).min()
            nearestindices = np.where(distvector == distmin)[0]
            nearestatom = nearestindices[0]

            self.connectivity[atom] = (atom,nearestatom)
            distances.append(distmin)

            # Compute Angles
            if atom >= 2:
                atms = [0, 0, 0]
                atms[0] = atom
                atms[1] = self.connectivity[atms[0]][1]
                atms[2] = self.connectivity[atms[1]][1]
                if atms[2] == atms[1]:
                    for idx in range(1, len(self.connectivity[:atom])):
                        if self.connectivity[idx][1] in atms and not idx in atms:
                            atms[2] = idx
                            break

                self.angleconnectivity[atom] = (atms[0], atms[1], atms[2])
                indices = [atms[0], atms[1], atms[2]]
                angles.append(self.get_angle(eq_geom, indices))

            # Compute Dihedral Angles
            if atom >= 3:
                atms = [0, 0, 0, 0]
                atms[0] = atom
                atms[1] = self.connectivity[atms[0]][1]
                atms[2] = self.angleconnectivity[atms[0]][2]
                atms[3] = self.angleconnectivity[atms[1]][2]
                if atms[3] in atms[:3]:
                    for idx in range(1, len(self.connectivity[:atom])):
                        if self.connectivity[idx][1] in atms and not idx in atms:
                            atms[3] = idx
                            break
                
                indices = [atms[0], atms[1], atms[2], atms[3]]
                dihedrals.append(self.get_dihedral(eq_geom, indices))
#                if math.isnan(self.dihedrals[atom]):
#                    dihedrals.append(0.0)

                self.dihedralconnectivity[atom] = (atms[0], atms[1], atms[2], atms[3])

        self.zmat_ref_geom = np.array(distances + angles + dihedrals, dtype=np.float64)

        self.connectivity = self.connectivity[1:]
        self.angleconnectivity = self.angleconnectivity[2:]
        self.dihedralconnectivity = self.dihedralconnectivity[3:]
        
        n_samples = all_geoms.shape[0]
        distances = np.empty((n_samples,len(self.connectivity)), dtype=np.float64)
        angles = np.empty((n_samples,len(self.angleconnectivity)), dtype=np.float64)
        dihedrals = np.empty((n_samples,len(self.dihedralconnectivity)), dtype=np.float64)

        # Here we calculate the elements of the Z-Matrix for the whole dataset
        # using the atom indices obtained from the reference geometry
        for i, geom in enumerate(all_geoms):

            for j, idx in enumerate(self.connectivity):
                distances[i][j] = self.get_distance(geom, idx)

            for j, idx in enumerate(self.angleconnectivity):
                angles[i][j] = self.get_angle(geom, idx)

            for j, idx in enumerate(self.dihedralconnectivity):
                dihedrals[i][j] = self.get_dihedral(geom, idx)

        col_names = list()
        # Column labels for bond distances
        col_names += ['r' + ''.join(map(str,np.array(idx)+1)) for idx in self.connectivity]
        # Column labels for angles
        col_names += ['a' + ''.join(map(str,np.array(idx)+1)) for idx in self.angleconnectivity]
        # Column labels for dihedrals 
        col_names += ['d' + ''.join(map(str,np.array(idx)+1)) for idx in self.dihedralconnectivity]

        df_zmat_refgeom = pd.DataFrame(self.zmat_ref_geom.reshape(1,-1), columns=col_names)

        zmat_all = np.hstack((distances, angles, dihedrals))

        if delta:
            zmat_all = zmat_all - self.zmat_ref_geom
            # Convert angles from degree to radians
            select_angles = len(self.connectivity) + 1
            zmat_all[:, select_angles:] *= (np.pi / 180)
            
        df = pd.DataFrame(zmat_all, columns=col_names)

        if save_csv:
            df.to_csv("zmatrix_all_geoms.csv", index=False)

        return df

    def transform(self,zmat_data,funct) -> pd.core.frame.DataFrame:
        """
        Apply a non-linear transformation to the delta Z-Matrix dataset.

        Implemented functions:
          -sigmoid: retuns a dataframe with values ranging from 0 to 1
          -tanh: hyperbolic tangent for bond distances and cosine for angles,
                 returns a dataframe with values in the range [-1,1]  
        """
        # Normalization functions
        sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
        tanh = np.vectorize(lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))

        col_names = zmat_data.columns.tolist()
        zmat_data = zmat_data.values

        if funct.lower() == 'sigmoid':    
            zmat_data = sigmoid(zmat_data)

        if funct.lower() == 'tanh':
            bond_features = len(self.connectivity)
            angle_features = bond_features + 1
            zmat_data[:,:bond_features] = tanh(zmat_data[:,:bond_features])
            zmat_data[:,angle_features:] = np.cos(zmat_data[:,angle_features:])

        df = pd.DataFrame(zmat_data, columns=col_names)

        return df
