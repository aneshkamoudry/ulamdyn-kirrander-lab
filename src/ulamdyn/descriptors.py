"""Classes and methods used to generate ML descriptors from molecular geometries."""
# Author: Max Pinheiro Jr <maxjr82@gmail.com> ; Bidhan Chandra Garain <bidhanchandragarain@gmail.com>
# Date: 03/10/2021

import os
from itertools import combinations
import numpy as np
from dscribe.descriptors import SOAP
from tqdm import tqdm
import ase

try:
    import modin.pandas as pd

except ModuleNotFoundError:
    import pandas as pd

from ulamdyn.data_loader import GetCoords
from ulamdyn.nx_utils import *

__all__ = ["R2", "ZMatrix", "RingParams", "SOAPDescriptor"]

filedir = os.path.dirname(__file__)


class R2(GetCoords):
    """Class used to convert the XYZ coordinates of molecular geometries into R2-type of descriptors.

    The R2 descriptor is defined as the (flattened) matrix of all pairwise Euclidean distances
    between all atoms in the molecule. Since the matrix is symmetric with respect to the interchange
    of atom indices (i.e., Dij = Dji) only the lower triangular portion of the R2 matrix will be
    outputed in the final data set.

    This class also provides a method to compute other molecular descriptors derived from the R2
    distance matrix. They are:

    - inverse R2 -> defined as :math:`1/R_{ij}`, similarly to the Coulomb matrix descriptor.
    - delta R2 -> difference between the R2 vector of the current geometry in time *t* and the equivalent R2 vector of a reference geometry (typically the ground-state geometry), :math:`R_{ij}(t) - R_{ij}(ref)`.
    - RE -> R2 vector normalized relative to equilibrium geometry, :math:`R_{ij}(eq)/R_{ij}(t)`. For more details, check the reference `J. Chem. Phys. 146, 244108 (2017) <https://aip.scitation.org/doi/10.1063/1.4989536>`_.

    """

    __slots__ = [
        "r2_ref_geom",
        "r2_descriptor",
        "mass_weights",
        "norm_factor",
        "apply_to_delta",
        "variant",
    ]

    def __str__(self) -> str:
        """Provide a string representation of the class.

        :return: Short description of the class functionality.
        :rtype: str
        """
        return "Generator of R2-based descriptors from molecular geometries."

    def __init__(self, all_geoms=None, use_mwc=False) -> None:
        """Class initializer.

        :param all_geoms: contain geometry information either collected from available
                          MD trajectories as an object of the :class:`~ulamdyn.GetCoords
                          or given as a tensor with all stacked XYZ coordinates in the
                          shape (n_samples, n_atoms, 3). if not provided, the method
                          :meth:`~ulamdyn.GetCoords.read_all_trajs` will be called to
                          load all geometries and build the tensor. Defaults to None.
        :type all_geoms: ulamdyn.GetCoords | numpy.ndarray
        :param use_mwc: If True, use mass weighted coordinates to compute the descriptors,
                        defaults to False.
        :type use_mwc: bool
        """
        super().__init__()
        if all_geoms is None:
            self.read_all_trajs()
        elif isinstance(all_geoms, GetCoords):
            self.xyz = all_geoms.xyz
            self.traj_time = all_geoms.traj_time
        elif isinstance(all_geoms, np.ndarray):
            self.xyz = all_geoms

        self.r2_ref_geom = None
        self.r2_descriptor = None
        self.mass_weights = None
        self.norm_factor = 1
        self.variant = None
        self.apply_to_delta = None
        if use_mwc:
            self.mass_weights = self._get_mass_weights()
            self.norm_factor = self._mass_norm_factor(self.mass_weights)

    def _get_mass_weights(self) -> np.ndarray:
        traj = list(self.trajectories)[0]
        _, atom_mass = get_labels_masses(traj)
        atom_mass = atom_mass.reshape(-1, 1)
        return atom_mass

    @staticmethod
    def _mass_norm_factor(mass: np.ndarray) -> np.ndarray:
        n_atoms = len(mass)
        m_pairs = np.tril_indices(n_atoms, -1)
        m_pairs = np.column_stack((m_pairs[1], m_pairs[0]))
        norm_factor = [np.sqrt(mass[i] * mass[j]) ** (-1) for i, j in m_pairs]
        norm_factor = np.array(norm_factor, dtype=np.float64).T
        return norm_factor

    @staticmethod
    def xyz_to_distances(xyz_matrix: np.ndarray) -> np.ndarray:
        """Calculate the pairwise distance matrix for a given XYZ geometry.

        :param xyz_matrix: Cartesian coordinates of a molecular structure given as a
                           matrix of shape (n_atoms, 3).
        :type xyz_matrix: numpy.ndarray
        :return: vector of size :math:`n_{atoms} (n_{atoms} - 1)/2` containing the
                 lower triangular portion of the R2 matrix.
        :rtype: numpy.ndarray
        """
        n_atoms = len(xyz_matrix)
        distance_matrix = np.zeros((n_atoms, n_atoms))

        for i, j in combinations(range(len(xyz_matrix)), 2):
            # mass-weighted dist:
            # norm((m[i]*xyz_matrix[i] - m[j]*xyz_matrix[j])/sqrt(m[i]*m[j])
            R = np.linalg.norm(xyz_matrix[i] - xyz_matrix[j])
            distance_matrix[j, i] = R

        r2_vector = distance_matrix[np.tril_indices(len(distance_matrix), -1)]

        return r2_vector

    @staticmethod
    def __build_r2_matrix(n, arr):
        m = np.zeros([n, n], dtype=np.float64)
        ind_lower_triang = np.tril_indices(n, -1)
        ind_pair = [(i, j) for i, j in zip(ind_lower_triang[0], ind_lower_triang[1])]
        for n, pair in enumerate(ind_pair):
            i, j = pair
            m[i, j] = arr[n]
            m[j, i] = arr[n]
        return m

    @staticmethod
    def reconstruct_xyz(r2_vec: np.ndarray) -> np.ndarray:
        """Reconstruct the XYZ coordinates from the pairwise atom distance vector.

        :param r2_vec: Numpy 2D array of shape (n_samples, n_atoms * (n_atoms - 1)/2)
                       that contains stacked R2 vectors.
        :type r2_vec: numpy.ndarray
        :return: Numpy 3D array of shape (n_samples, n_atoms, 3) containing the molecular
                 geometries transformed back into the Cartesian coordinate space.
        :rtype: numpy.ndarray
        """
        n_samples, dim = r2_vec.shape
        n_atoms = np.int((1 + np.sqrt(1 + 8 * dim)) / 2)

        xyz_reconstructed = np.zeros([n_samples, n_atoms, 3], dtype=np.float64)
        for n, vec in enumerate(r2_vec):
            d = R2.__build_r2_matrix(n_atoms, vec)
            # Multidimensional scaling algorithm
            E = -0.5 * d**2

            # Use mat to generate column and row means.
            Er = np.mat(np.mean(E, 1))
            Es = np.mat(np.mean(E, 0))

            # From Principles of Multivariate Analysis: A User's Perspective (page 107).
            F = np.array(E - np.transpose(Er) - Es + np.mean(E))

            [U, S, V] = np.linalg.svd(F)
            Y = U * np.sqrt(S)

            coords = np.asarray(Y[:, 0:3])
            xyz_reconstructed[n] = coords
        return xyz_reconstructed

    @staticmethod
    def _transform(delta_r2_data, funct, inverse=False):
        """Apply a non-linear transformation to the delta R2 descriptor.

        Functions implemented:
        ----------------------
          -sigmoid: retuns a dataframe with values ranging from 0 to 1
          -tanh: apply the hyperbolic tangent function to all features,
                 returning a dataframe with values in the range [-1,1]

        :param delta_r2_data: dataset will all stacked delta R2 descriptors.
        :type transf_dr2_data: numpy.ndarray
        :param funct: select function for nonlinear transformation.
        :type funct: str
        :param inverse: calculate the inverse transformation for funct.
        :type inverse: bool
        """
        # Normalization functions
        sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
        tanh = np.vectorize(
            lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        )
        # Inverse transformation for the normalization functions
        inv_sigmoid = np.vectorize(lambda x: np.log(x / (1 - x)))
        inv_tanh = np.vectorize(np.arctanh)

        funct = funct.strip().replace(" ", "").lower()
        if inverse:
            funct = "inv_" + funct
        funct = eval(funct)
        trans_dr2_data = funct(delta_r2_data)

        return trans_dr2_data

    def _derived_model(self, variant: str) -> None:
        if variant == "inv-R2":
            self.r2_descriptor = 1 / self.r2_descriptor
        else:
            self.read_eq_geom()
            eq_geom = self.eq_xyz.copy()
            if self.mass_weights is not None:
                eq_geom *= self.mass_weights
                self.norm_factor = self.norm_factor.ravel()

            self.r2_ref_geom = self.xyz_to_distances(eq_geom)
            self.r2_ref_geom *= self.norm_factor

            if variant == "delta-R2":
                self.r2_descriptor = self.r2_descriptor - self.r2_ref_geom

            if variant == "RE":
                self.r2_descriptor = self.r2_ref_geom / self.r2_descriptor

    def inverse_transform(self, descriptor):
        """Apply all transformations in reverse order to recover the original R2 vector."""

        if self.variant == "inv-R2":
            descriptor = 1 / descriptor
        if self.variant == "RE":
            descriptor = (1 / descriptor) * self.r2_ref_geom
        if self.variant == "delta-R2":
            if self.apply_to_delta is not None:
                descriptor = self._transform(
                    descriptor, self.apply_to_delta, inverse=True
                )
            descriptor = descriptor + self.r2_ref_geom

        descriptor *= 1 / self.norm_factor
        return descriptor

    def build_descriptor(self, variant=None, apply_to_delta=None, save_csv=False):
        """Generate a dataframe with R2-based descriptors for all molecular geometries.

        :param variant: molecular representation derived from the R2 descriptor,
                        defaults to None.
        :type variant: str, optional
        :param apply_to_delta: select a nonlinear function to apply as a transformation (sigmoid
                          or hyperbolic tangent) on delta-R2 descriptors, defaults to None.
        :type apply_to_delta: str, optional
        :param save_csv: if True export the data set with all calculated descriptors in
                         a csv format with name all_geoms_r2.csv, defaults to False.
        :type save_csv: bool, optional
        :return: a dataframe object of shape (n_samples, n_atoms * (n_atoms - 1)/2), where
                 each row is a vector with the R2-based descriptor computed for a given
                 molecular geometry.
        :rtype: pandas.DataFrame | modin.pandas.dataframe.DataFrame
        """
        all_geoms = self.xyz
        n_samples, n_atoms, _ = all_geoms.shape
        id_atom_pairs = np.tril_indices(n_atoms, -1)
        n_features = len(id_atom_pairs[0])

        if self.mass_weights is not None:
            all_geoms *= self.mass_weights

        self.r2_descriptor = np.empty((n_samples, n_features), dtype=np.float64)

        for i, xyz in enumerate(all_geoms):
            d = self.xyz_to_distances(xyz)
            self.r2_descriptor[i] = d

        self.r2_descriptor *= self.norm_factor

        func = lambda x, y: "r" + "".join(sorted([str(y + 1), str(x + 1)], key=int))
        col_names = list(map(func, id_atom_pairs[0], id_atom_pairs[1]))

        if variant in ["inv-R2", "delta-R2", "RE"]:
            self._derived_model(variant)
            self.variant = variant
            if (variant == "delta-R2") and (apply_to_delta is not None):
                self.r2_descriptor = self._transform(self.r2_descriptor, apply_to_delta)
                self.apply_to_delta = apply_to_delta

        df_r2 = pd.DataFrame(self.r2_descriptor, columns=col_names)
        if self.traj_time is not None:
            df_r2.insert(0, "TRAJ", self.traj_time[:, 0])
            df_r2.insert(1, "time", self.traj_time[:, 1])
            df_r2["TRAJ"] = df_r2["TRAJ"].astype("int32")

        if save_csv:
            df_r2.to_csv("all_geoms_r2.csv", index=False)

        return df_r2


class ZMatrix(GetCoords):
    """Class used to generate molecular descriptors using internal coordinates (Z-Matrix).

    This class does not require arguments in its constructor. All quantities related to
    distances are given in angstrom, while the features derived from angles are provided
    in degrees.

    In addition to the standard Z-Matrix, the class also provides a method to compute other
    variants of the Z-Matrix molecular descriptors:

    - delta Z-Matrix -> difference between the Z-Matrix representation of the current geometry in time *t* and the Z-Matrix of a reference geometry.
    - tanh Z-Matrix -> hyperbolic tangent transformation on all features of delta Z-Matrix.
    - sig Z-Matrix -> sigmoid transformation on all features of delta Z-Matrix.

    Data attributes:
    ----------------
       ``distancematrix`` (numpy.ndarray): stores the full matrix of bond distances for all geometries.\n
       ``connectivity`` (list): indices of connected atoms based on a distance criterion of proximity.\n
       ``angleconnectivity`` (list): indices of three neighboring atoms to compute angles.\n
       ``dihedralconnectivity`` (list): four indices of neighboring atoms to calculate dihedrals.\n
       ``zmat_ref_geom`` (numpy.ndarray): stores the Z-matrix calculated for the reference geometry.

    """

    # Defining slots to optimize performance (RAM):
    __slots__ = [
        "distancematrix",
        "connectivity",
        "angleconnectivity",
        "dihedralconnectivity",
        "zmat_ref_geom",
    ]

    def __str__(self) -> str:
        """Provide a string representation of the class.

        :return: Short description of the class functionality.
        :rtype: str
        """
        return "Generator of Z-Matrix descriptors from molecular geometries."

    def __init__(self, all_geoms=None) -> None:
        """Class initializer.

        :param all_geoms: contain geometry information either collected from available
                          MD trajectories as an object of the :class:`~ulamdyn.GetCoords
                          or given as a tensor with all stacked XYZ coordinates in the
                          shape (n_samples, n_atoms, 3). if not provided, the method
                          :meth:`~ulamdyn.GetCoords.read_all_trajs` will be called to
                          load all geometries and build the tensor. Defaults to None.
        :type all_geoms: ulamdyn.GetCoords | numpy.ndarray
        """
        super().__init__()
        if all_geoms is None:
            self.read_all_trajs()
        elif isinstance(all_geoms, GetCoords):
            self.xyz = all_geoms.xyz
            self.traj_time = all_geoms.traj_time
        elif isinstance(all_geoms, np.ndarray):
            self.xyz = all_geoms

        self.distancematrix = None

        # Internal Coordinate Connectivity
        self.connectivity = None
        self.angleconnectivity = None
        self.dihedralconnectivity = None

        # Z-Matrix of the reference molecular geometry
        self.zmat_ref_geom = None

    @staticmethod
    def get_distance(geom: np.ndarray, idx_atoms: list) -> np.float64:
        """Calculate the Euclidean distance between a pair of atoms.

        :param geom: matrix of shape (natoms, 3) storing the XYZ coordinates of a single molecule.
        :type geom: numpy.ndarray
        :param idx_atoms: a pair of indices corresponding to the atoms for which the distance will
                          be calculated.
        :type idx_atoms: list
        :return: Euclidean distance (in Angstrom) between two selected atoms.
        :rtype: numpy.float
        """
        i, j = idx_atoms
        vec = geom[j] - geom[i]
        dist = np.linalg.norm(vec)

        return dist

    @staticmethod
    def get_angle(geom: np.ndarray, idx_atoms: list) -> np.float64:
        """Calculate the angle formed by three selected atoms.

        :param geom: matrix of shape (natoms, 3) storing the XYZ coordinates of a single molecule.
        :type geom: numpy.ndarray
        :param idx_atoms: a list of three indices corresponding to the atoms for which the angle
                          will be calculated.
        :type idx_atoms: list
        :return: angle (in degrees) between three selected atoms.
        :rtype: numpy.float
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
    def get_dihedral(geom: np.ndarray, idx_atoms: list) -> np.float64:
        """Calculate the dihedral angle formed by four selected atoms.

        :param geom: matrix of shape (natoms, 3) storing the XYZ coordinates of a single molecule.
        :type geom: numpy.ndarray
        :param idx_atoms: a list of four indices to select the atoms for which the dihedral angle
                          will be calculated.
        :type idx_atoms: list
        :return: dihedral angle (in degrees) formed by four specified atoms.
        :rtype: numpy.float
        """
        if not isinstance(idx_atoms, list):
            idx_atoms = list(idx_atoms)

        (p0, p1, p2, p3) = geom[idx_atoms]

        b0 = -1.0 * (p1 - p0)
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
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1

        # angle between v and w in a plane is the torsion angle
        # v and w may not be normalized but that's fine since tan is y/x
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)

        phi = np.degrees(np.arctan2(y, x))

        return phi

    @staticmethod
    def get_bending(geom: np.ndarray, idx_atoms: list) -> np.float64:
        """Calculate the bending angle between two planes of the molecule.

        This method is particularly useful to describe large out-of-plane distortions in the
        molecular structure that involves more than four atoms. The bending angle is calculated
        by first defining two vectors each one perpendicular to different molecular planes
        formed by two sets of three atoms. Then, the angle between the two vectors is obtained
        by calculating the inverse cosine of the scalar product between these vectors.

        .. note:: By default, the bending angle is not used to construct the Z-Matrix descriptor.
                  It can be used to construct an augmented version of the Z-Matrix that better
                  captures changes in the molecular structure during the dynamics.

        :param geom: matrix of shape (natoms, 3) storing the XYZ coordinates of a single molecule.
        :type geom: numpy.ndarray
        :param idx_atoms: a list of lists with three atom indices in each, used to define two
                          molecular planes for which the angle will be calculated.
        :type idx_atoms: list
        :return: bending angle (in degrees) defined by six specified atoms.
        :rtype: np.float
        """
        idx_ring1, idx_ring2 = idx_atoms

        # Calculate the vector perpendicular to the plan of the first ring
        v1_ring1 = (geom[idx_ring1[2]] - geom[idx_ring1[0]]) / 2
        v2_ring1 = (geom[idx_ring1[1]] - geom[idx_ring1[0]]) / 2
        normal_to_ring1 = np.cross(v2_ring1, v1_ring1)
        normal_to_ring1 /= np.linalg.norm(normal_to_ring1)

        # Calculate the vector perpendicular to the plan of the second ring
        v1_ring2 = (geom[idx_ring2[2]] - geom[idx_ring2[0]]) / 2
        v2_ring2 = (geom[idx_ring2[1]] - geom[idx_ring2[0]]) / 2
        normal_to_ring2 = np.cross(v2_ring2, v1_ring2)
        normal_to_ring2 /= np.linalg.norm(normal_to_ring2)

        # Calculate the angle between the two normal vectors
        cos_theta = normal_to_ring1.dot(normal_to_ring2)
        omega = np.arccos(cos_theta)
        omega = np.degrees(omega)

        return omega

    def _build_distance_matrix(self, xyz_matrix: np.ndarray):
        n_atoms = xyz_matrix.shape[0]
        self.distancematrix = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in [x for x in range(n_atoms) if x > i]:
                self.distancematrix[i][j] = np.linalg.norm(
                    xyz_matrix[i] - xyz_matrix[j]
                )
                self.distancematrix[j][i] = self.distancematrix[i][j]

    @property
    def _gen_column_labels(self):
        col_names = list()
        # Column labels for bond distances
        col_names += [
            "r" + "".join(map(str, np.array(idx) + 1)) for idx in self.connectivity
        ]
        # Column labels for angles
        col_names += [
            "a" + "".join(map(str, np.array(idx) + 1)) for idx in self.angleconnectivity
        ]
        # Column labels for dihedrals
        col_names += [
            "d" + "".join(map(str, np.array(idx) + 1))
            for idx in self.dihedralconnectivity
        ]
        return col_names

    def _transform(self, zmat_data, funct):
        """Apply a non-linear transformation to the delta Z-Matrix dataset.

        Functions implemented:
        ----------------------
          -sigmoid: retuns a dataframe with values ranging from 0 to 1
          -tanh: hyperbolic tangent for bond distances and cosine for angles,
                 returns a dataframe with values in the range [-1,1]

        :param zmat_data: dataset will delta Z-matrix descriptors.
        :type zmat_data: numpy.ndarray
        """
        # Normalization functions
        sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
        tanh = np.vectorize(
            lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        )

        bond_features = len(self.connectivity)
        angle_features = bond_features + 1

        if funct.lower() == "sigmoid":
            zmat_data[:, :bond_features] = sigmoid(zmat_data[:, :bond_features])
            zmat_data[:, angle_features:] = 1 - np.cos(zmat_data[:, angle_features:])

        if funct.lower() == "tanh":
            zmat_data[:, :bond_features] = tanh(zmat_data[:, :bond_features])
            zmat_data[:, angle_features:] = np.cos(zmat_data[:, angle_features:])

        return zmat_data

    def build_descriptor(self, delta=False, apply_to_delta=None, save_csv=False):
        """Construct the standard Z-Matrix descriptor and other variants.

        .. note:: By default, the algorithm will calculate the three main components of the
                  Z-Matrix: *bond distances*, *angles* and *dihedrals*. An augmented version
                  of the Z-Matrix descriptor can be also obtained by calculating additional
                  distances, angles, dihedrals and/or bending angles using the methods provided
                  in the class.

        :param delta: if True, the Z_matrix feature vector of each geometry will be subtracted
                      from the Z_Matrix of the reference geometry, defaults to False.
        :type delta: bool, optional
        :param apply_to_delta: select a nonlinear function to apply as a transformation (sigmoid
                          or hyperbolic tangent) on the delta Z-matrix, defaults to None.
        :type apply_to_delta: str, optional
        :param save_csv: if true save a single csv file named all_geoms_zmatrix.csv containing
                         the Z-Matrix descriptors computed for all geometries available the MD
                         trajectories., defaults to False.
        :type save_csv: bool, optional
        :return: a dataframe object with the (flattened) Z-Matrix descriptors stacked for all
                 MD geometries.
        :rtype: pandas.DataFrame
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
        self.connectivity = [(0, 0) for _ in range(0, n_atoms)]
        self.angleconnectivity = [(0, 0, 0) for _ in range(0, n_atoms)]
        self.dihedralconnectivity = [(0, 0, 0, 0) for _ in range(0, n_atoms)]

        distances = list()
        angles = list()
        dihedrals = list()

        # This first loop goes over the atoms of the reference geometry
        # to obtain the Z-Matrix and all indices of "connected" atoms
        for atom in range(1, n_atoms):
            # For current atom, find the nearest atom among previous atoms
            distvector = self.distancematrix[atom][:atom]
            distmin = np.array(distvector[np.nonzero(distvector)]).min()
            nearestindices = np.where(distvector == distmin)[0]
            nearestatom = nearestindices[0]

            self.connectivity[atom] = (atom, nearestatom)
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

        all_geoms = self.xyz
        n_samples = all_geoms.shape[0]
        distances = np.empty((n_samples, len(self.connectivity)), dtype=np.float64)
        angles = np.empty((n_samples, len(self.angleconnectivity)), dtype=np.float64)
        dihedrals = np.empty(
            (n_samples, len(self.dihedralconnectivity)), dtype=np.float64
        )

        # Here we calculate the elements of the Z-Matrix for the whole dataset
        # using the atom indices obtained from the reference geometry
        for i, geom in enumerate(all_geoms):
            for j, idx in enumerate(self.connectivity):
                distances[i][j] = self.get_distance(geom, idx)

            for j, idx in enumerate(self.angleconnectivity):
                angles[i][j] = self.get_angle(geom, idx)

            for j, idx in enumerate(self.dihedralconnectivity):
                dihedrals[i][j] = self.get_dihedral(geom, idx)

        zmat_all = np.hstack((distances, angles, dihedrals))

        if delta:
            zmat_all = zmat_all - self.zmat_ref_geom
            # Convert angles from degree to radians
            select_angles = len(self.connectivity) + 1
            zmat_all[:, select_angles:] *= np.pi / 180

            if apply_to_delta:
                zmat_all = self._transform(zmat_all, apply_to_delta)

        col_names = self._gen_column_labels
        df = pd.DataFrame(zmat_all, columns=col_names)
        if self.traj_time is not None:
            df.insert(0, "TRAJ", self.traj_time[:, 0])
            df.insert(1, "time", self.traj_time[:, 1])
            df["TRAJ"] = df["TRAJ"].astype("int32")

        if save_csv:
            df.to_csv("all_geoms_zmatrix.csv", index=False)

        return df


class RingParams(GetCoords):
    """Class used to the Cremer-Pople parameters from ring coordinates."""

    def __str__(self) -> str:
        """Provide a string representation of the class.

        :return: Short description of the class functionality.
        :rtype: str
        """
        return "Cremer-Pople parameter calculator for cyclic molecular fragments."

    def __init__(self, ring_atom_ind: list, ring_coords: np.ndarray = None) -> None:
        """Class initializer."""
        super().__init__()
        # List of atom indices that defines the ring
        self.ring_indices = [i - 1 for i in ring_atom_ind]
        self.ring_coords = ring_coords
        # Number of atoms in the ring
        self.ring_size = len(ring_atom_ind)

    def _fixzero(self, x) -> np.ndarray:
        x_ = np.array([0.0]) if np.allclose(0, x, rtol=1e-06, atol=1e-08) else x
        return x_

    @property
    def ring_coords(self) -> np.ndarray:
        """Filter the XYZ coordinates corresponding to the selected ring and translate the coordinates to the ring center."""
        return self._ring_coords

    @ring_coords.setter
    def ring_coords(self, coordinates: np.ndarray) -> None:
        if coordinates is not None:
            if coordinates.ndim == 2:
                coordinates = np.expand_dims(coordinates, axis=0)
            ring_coords = coordinates
        else:
            self.read_all_trajs()
            ring_coords = self.xyz[:, self.ring_indices]
        self._ring_coords = ring_coords - ring_coords.mean(axis=(1,), keepdims=True)

    def _cp_to_polar(self, pucker_coords) -> dict:
        Q = np.sqrt(np.power(pucker_coords[:, :2], 2).sum(axis=1))
        theta = np.arctan(pucker_coords[:, 0] / pucker_coords[:, 1])
        theta = theta * (180 / np.pi)
        # Shift the negative angles to stay within the 360 deg circle
        theta = np.where(theta < 0, theta + 360, theta)
        phi = pucker_coords[:, -1]
        polar_coords = {"Q": Q, "theta": theta, "phi": phi}
        return polar_coords

    def _get_ang_components(self, z, rs, m) -> tuple:
        cos_term = [np.dot(z, np.cos(2 * np.pi * k * np.arange(0, rs) / rs)) for k in m]
        sin_term = [np.dot(z, np.sin(2 * np.pi * k * np.arange(0, rs) / rs)) for k in m]
        qcos = self._fixzero(np.sqrt(2 / rs) * np.array(cos_term))
        qsin = self._fixzero(-np.sqrt(2 / rs) * np.array(sin_term))
        return (qcos, qsin)

    def displacement(self, coords: np.ndarray):
        """Calculate the ring displacement (z)"""
        rs = self.ring_size
        R1 = np.dot(np.sin(2 * np.pi * np.arange(0, rs) / rs), coords)
        R2 = np.dot(np.cos(2 * np.pi * np.arange(0, rs) / rs), coords)
        cross_product = np.cross(R1, R2)
        normal_vec = cross_product / np.linalg.norm(cross_product)
        z = np.dot(coords, normal_vec)
        return z

    def get_pucker_coords(self, coords: np.ndarray) -> np.ndarray:
        """Calculate the Cremer-Pople puckering parameters for one ring."""

        rs = self.ring_size
        z = self.displacement(coords)
        if rs > 4 and rs <= 20:
            if rs % 2 == 0:
                m = range(2, int((rs / 2)))
                qcos, qsin = self._get_ang_components(z, rs, m)
                q = np.sqrt(qsin**2 + qcos**2)
                amplitude = np.append(
                    q,
                    (1 / np.sqrt(rs))
                    * np.dot(z, np.cos(np.arange(0, rs) * np.pi)).sum(),
                )
                angle = np.arctan2(qsin, qcos)
            else:
                m = range(2, int((rs - 1) / 2) + 1)
                qcos, qsin = self._get_ang_components(z, rs, m)
                amplitude = np.sqrt(qsin**2 + qcos**2)
                angle = np.arctan2(qsin, qcos)
        else:
            print("ERROR:")
            print("Ring size not supported!")
            print("The number of atoms should be 4 < n_atoms <= 20.")
        # Convert from radian to degree
        if angle < 0.0:
            angle += 2 * np.pi
        angle = angle * (180 / np.pi)
        cppar = np.concatenate([amplitude, angle])
        return cppar

    @staticmethod
    def _reduce_angle(*args):
        if len(args) == 1:
            phi = args[0]
            theta = None
        elif len(args) == 2:
            phi, theta = args

        # Reduce angle to 0 <= phi <= 360
        while phi < 0:
            phi += 360
        while phi > 360:
            phi -= 360

        if theta is not None:
            # Reduce theta
            if (theta < 0) or (theta > 360):
                sign = np.sign(theta)
                n360 = max(1, int(abs(theta) / 360))
                theta = theta - sign * n360 * 360
            if theta > 180:
                theta = theta - 180
        return (phi, theta)

    def get_conf_5memb(self, phi) -> str:
        """Determine the class of a 5-membered ring deformation based on the CP parameters."""
        # Reduce angle to 0 <= phi <= 360
        phi, _ = self._reduce_angle(phi)
        # while phi < 0:
        #    phi += 360
        # while phi > 360:
        #    phi -= 360

        invert = False
        if phi >= 180:
            phi -= 180
            invert = True

        classes = [
            "1E",
            "2T1",
            "2E",
            "2T3",
            "3E",
            "4T3",
            "4E",
            "4T5",
            "E5",
            "1T5",
            "1E",
        ]

        conf = classes[int(phi / 18)]
        if invert:
            conf = conf[::-1]
        return conf

    def get_conf_6memb(self, theta, phi) -> str:
        """Determine the conformation class for a 6-membered ring based on the CP parameters."""
        sqrt2 = np.sqrt(2.0)
        sqrt32 = np.sqrt(1.5)

        phi, theta = self._reduce_angle(phi, theta)
        # DEG -> RAD
        phi = np.radians(phi)
        theta = np.radians(theta)

        # Phi test
        n_test = 6.0 / np.pi * phi
        trunc_n = int(n_test)
        remainder_n = n_test - trunc_n
        if remainder_n < 0.5:
            n_final = trunc_n
        elif remainder_n >= 0.5:
            n_final = trunc_n + 1
        if n_final % 2:
            phi_class = "HST"
        else:
            phi_class = "EBE"

        # Theta test
        L_C = 0.0
        L_C1 = np.pi
        if phi_class == "EBE":
            L_E1 = np.arctan(sqrt2)
            L_B = np.pi / 2.0
            L_E2 = np.pi + np.arctan(-sqrt2)
            l0 = (L_C + L_E1) / 2.0
            l1 = (L_E1 + L_B) / 2.0
            l2 = (L_B + L_E2) / 2.0
            l3 = (L_E2 + L_C1) / 2.0
            if theta >= L_C and theta < l0:
                class_ = "C"
            elif theta >= l0 and theta < l1:
                class_ = "E1"
            elif theta >= l1 and theta < l2:
                class_ = "B"
            elif theta >= l2 and theta < l3:
                class_ = "E2"
            elif theta >= l3 and theta <= L_C1:
                class_ = "C"
            else:
                raise ValueError(
                    f"ERROR: theta = {np.degrees(theta)}deg: out of limits."
                )
            if class_ != "C":
                class_ = self._class_even_6memb(class_, n_final)
        if phi_class == "HST":
            L_H1 = np.arctan(sqrt32)
            L_S1 = np.arctan(1.0 + sqrt2)
            L_T = np.pi / 2.0
            L_S2 = np.pi + np.arctan(-(1.0 + sqrt2))
            L_H2 = np.pi + np.arctan(-sqrt32)
            l0 = (L_C + L_H1) / 2.0
            l1 = (L_H1 + L_S1) / 2.0
            l2 = (L_S1 + L_T) / 2.0
            l3 = (L_T + L_S2) / 2.0
            l4 = (L_S2 + L_H2) / 2.0
            l5 = (L_H2 + L_C1) / 2.0
            if theta >= L_C and theta < l0:
                class_ = "C"
            elif theta >= l0 and theta < l1:
                class_ = "H1"
            elif theta >= l1 and theta < l2:
                class_ = "S1"
            elif theta >= l2 and theta < l3:
                class_ = "T"
            elif theta >= l3 and theta < l4:
                class_ = "S2"
            elif theta >= l4 and theta < l5:
                class_ = "H2"
            elif theta >= l5 and theta <= L_C1:
                class_ = "C"
            else:
                raise ValueError(
                    f"ERROR: theta = {np.degrees(theta)}deg: out of limits."
                )
            if class_ != "C":
                class_ = self._class_odd_6memb(class_, n_final)
            if class_ == "C":
                class_ = self._class_c_6memb(class_, theta, phi)
        return class_

    @staticmethod
    def _class_even_6memb(class_var, n_final):
        cl = class_var[0]
        ind1 = (n_final + 2) // 2
        ind2 = ""
        if class_var == "E2":
            ind1 = ""

        if ind1 == "7":
            ind1 = 1

        if n_final == 0:
            if class_var in ["B", "E2"]:
                ind2 = 4
            class_var = str(ind1) + str(ind2) + cl
        elif n_final == 2:
            if class_var in ["B", "E2"]:
                ind2 = 5
            class_var = cl + str(ind1) + str(ind2)
        elif n_final == 4:
            if class_var in ["B", "E2"]:
                ind2 = 6
            class_var = str(ind2) + str(ind1) + cl
        elif n_final == 6:
            if class_var in ["B", "E2"]:
                ind2 = 1
            class_var = cl + str(ind1) + str(ind2)
        elif n_final == 8:
            if class_var in ["B", "E2"]:
                ind2 = 2
            class_var = str(ind2) + str(ind1) + cl
        elif n_final == 10:
            if class_var in ["B", "E2"]:
                ind2 = 3
            class_var = cl + str(ind1) + str(ind2)
        elif n_final == 12:
            if class_var in ["B", "E2"]:
                ind2 = 4
            class_var = str(ind2) + str(ind1) + cl
        else:
            raise ValueError("ERROR: phi angle is out of limits.")
        return class_var

    @staticmethod
    def _class_odd_6memb(class_var, n_final):
        if "H" in class_var:
            cl = "H"
        elif "S" in class_var:
            cl = "S"
        elif "T" in class_var:
            cl = "T"

        if n_final == 1:
            if class_var in ["H1", "S1"]:
                ind1 = 1
                ind2 = 2
            elif class_var in ["H2", "S2"]:
                ind1 = 4
                ind2 = 5
            elif class_var == "T":
                ind1 = 4
                ind2 = 2
        elif n_final == 3:
            if class_var in ["H1", "S1"]:
                ind1 = 3
                ind2 = 2
            elif class_var in ["H2", "S2"]:
                ind1 = 6
                ind2 = 5
            elif class_var == "T":
                ind1 = 6
                ind2 = 2
        elif n_final == 5:
            if class_var in ["H1", "S1"]:
                ind1 = 3
                ind2 = 4
            elif class_var in ["H2", "S2"]:
                ind1 = 6
                ind2 = 1
            elif class_var == "T":
                ind1 = 3
                ind2 = 1
        elif n_final == 7:
            if class_var in ["H1", "S1"]:
                ind1 = 5
                ind2 = 4
            elif class_var in ["H2", "S2"]:
                ind1 = 2
                ind2 = 1
            elif class_var == "T":
                ind1 = 2
                ind2 = 4
        elif n_final == 9:
            if class_var in ["H1", "S1"]:
                ind1 = 5
                ind2 = 6
            elif class_var in ["H2", "S2"]:
                ind1 = 2
                ind2 = 3
            elif class_var == "T":
                ind1 = 2
                ind2 = 6
        elif n_final == 11:
            if class_var in ["H1", "S1"]:
                ind1 = 1
                ind2 = 6
            elif class_var in ["H2", "S2"]:
                ind1 = 4
                ind2 = 3
            elif class_var == "T":
                ind1 = 1
                ind2 = 3
        else:
            raise ValueError("ERROR: phi angle is out of limits.")
        class_var = f"{ind1}{cl}{ind2}"
        return class_var

    @staticmethod
    def _class_c_6memb(class_var, theta, phi):
        ind1 = ""
        ind2 = ""
        if (0 <= phi) and (phi < np.pi / 6):
            ind1 = 1
            ind2 = 4
        if (np.pi / 6 <= phi) and (phi < np.pi / 2):
            ind1 = 2
            ind2 = 5
        if (np.pi / 2 <= phi) and (phi < 5 * np.pi / 6):
            ind1 = 3
            ind2 = 6
        if (5 * np.pi / 6 <= phi) and (phi < 7 * np.pi / 6):
            ind1 = 1
            ind2 = 4
        if (7 * np.pi / 6 <= phi) and (phi < 3 * np.pi / 2):
            ind1 = 2
            ind2 = 5
        if (3 * np.pi / 2 <= phi) and (phi < 11 * np.pi / 6):
            ind1 = 3
            ind2 = 6
        if (11 * np.pi / 6 <= phi) and (phi < np.pi):
            ind1 = 1
            ind2 = 4
        if theta > np.pi / 2.0:
            indaux = ind1
            ind1 = ind2
            ind2 = indaux
        class_var = f"{ind1}{class_var}{ind2}"
        return class_var

    def build_dataframe(self, save_csv=False):
        """Construct a data frame containing the Cremer-Pople parameters of all collected geometries."""
        all_pucker_params = []
        append_pucker_params = all_pucker_params.append
        for xyz in self.ring_coords:
            cppar = self.get_pucker_coords(xyz)
            append_pucker_params(cppar)

        all_pucker_params = np.asarray(all_pucker_params, dtype=np.float64)
        if self.ring_size == 6:
            polar_coords = self._cp_to_polar(all_pucker_params)
            df = pd.DataFrame(polar_coords)
            func_vec = np.vectorize(self.get_conf_6memb)
            df["class"] = func_vec(df["theta"].values, df["phi"].values)
        else:
            n_ring_params = self.ring_size - 3
            col_names = ["q" + str(i + 1) for i in range(n_ring_params - 1)]
            col_names += ["phi"]
            df = pd.DataFrame(all_pucker_params, columns=col_names)
            func_vec = np.vectorize(self.get_conf_5memb)
            df["class"] = func_vec(df["phi"].values)
        # If the trajectory indices and time steps are available in the parent class (GetCoords),
        # these information will be added to the current dataframe.
        df = self._insert_traj_time(df)

        if save_csv:
            df.to_csv("all_ring_params.csv", index=False)

        return df

class SOAPDescriptor(GetCoords):
    """Class used to generate SOAP (Smooth Overlap of Atomic Positions) descriptors for molecular geometries.

    This class extends the GetCoords class, allowing it to handle molecular geometries and generate
    SOAP descriptors based on the atomic coordinates and species extracted from those geometries.
    """

    __slots__ = [
        "species",
        "soap",
        "atoms",
    ]

    def __str__(self) -> str:
        """Provide a string representation of the class.

        :return: Short description of the class functionality.
        :rtype: str
        """
        return "Generator of SOAP descriptors from molecular geometries."

    def __init__(self, all_geoms=None, atoms=None, r_cut=14, n_max=8, l_max=6, average='outer', rbf='polynomial') -> None:
        """Class initializer.

        :param all_geoms: Molecular geometries, either collected from available
                        MD trajectories as an object of the :class:`~ulamdyn.GetCoords`
                        or given as a tensor with all stacked XYZ coordinates in the
                        shape (n_samples, n_atoms, 3). If not provided, the method
                        :meth:`~ulamdyn.GetCoords.read_all_trajs` will be called to
                        load all geometries and build the tensor. Defaults to None.
        :type all_geoms: ulamdyn.GetCoords | numpy.ndarray
        :param atoms: List of atomic symbols corresponding to the atoms in the geometries.
        :type atoms: list[str]
        :param r_cut: Cutoff radius for local environment, defaults to 14.
        :type r_cut: float, optional
        :param n_max: Maximum radial basis functions, defaults to 8.
        :type n_max: int, optional
        :param l_max: Maximum degree of spherical harmonics, defaults to 6.
        :type l_max: int, optional
        :param average: The averaging mode over centre of interest. The valid options are
                        'outer', 'inner' and 'off'. Defaults to 'outer'.
                        'off': No averaging
                        'inner': Averaging over sites before summing up magnetic quantum numbers
                        'outer': Averaging over the power spectrum of different sites
        :type average: str, optional
        :param rbf: Type of radial basis function to use, defaults to 'polynomial'. The valid options are
                        'gto': Sphreical gaussian type orbitals
                        'polynomial': polynomial basis function
        :type rbf: str, optional
        """
        super().__init__()

        # Process geometries and atoms
        self._initialize_geometries_and_atoms(all_geoms, atoms)

        # Sort unique species
        self.species = list(set(self.atoms))
        self.species.sort()
        self.features_soap = None

        # Initialize SOAP descriptor
        self.soap = SOAP(
            species=self.species,
            periodic=False,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            average=average,
            compression={'mode': 'off'},
            rbf=rbf
        )

    def _initialize_geometries_and_atoms(self, all_geoms, atoms):
        """
        Helper function to initialize geometries and atom types.

        :param all_geoms: Molecular geometries, either collected from available
                        MD trajectories or given as a tensor.
        :type all_geoms: ulamdyn.GetCoords | numpy.ndarray
        :param atoms: List of atomic symbols corresponding to the atoms in the geometries.
        :type atoms: list[str]
        :raises ValueError: If `atoms` is not provided.
        """
        if all_geoms is None:
            self.read_all_trajs()
        elif isinstance(all_geoms, GetCoords):
            self.xyz = all_geoms.xyz
            self.traj_time = all_geoms.traj_time
        elif isinstance(all_geoms, np.ndarray):
            self.xyz = all_geoms

        if atoms is None:
            raise ValueError("Atom list must be provided.")

        self.atoms = atoms

    def create_features(self) -> pd.DataFrame:
        """Generate SOAP descriptors for the provided molecular geometries.

        :return: Array of SOAP descriptors for each molecular geometry.
        :rtype: numpy.ndarray
        """
        if self.xyz is None or self.atoms is None:
            raise ValueError("Molecular geometries (xyz) and atom types (atoms) must be provided.")

        mol = ase.Atoms(self.atoms, self.xyz[0])
        size_soap = len(self.soap.create(mol, n_jobs=-1))

        self.features_soap = np.zeros([len(self.xyz), size_soap], dtype=np.float64)

        for i in (range(len(self.xyz))):
            mol = ase.Atoms(self.atoms, self.xyz[i])
            self.features_soap[i, :] = self.soap.create(mol, n_jobs=-1)

        n_features = size_soap
        col_names = list(range(n_features))
        df_soap = pd.DataFrame(self.features_soap, columns=col_names)

        if self.traj_time is not None:
            df_soap.insert(0, "TRAJ", self.traj_time[:, 0])
            df_soap.insert(1, "time", self.traj_time[:, 1])
            df_soap["TRAJ"] = df_soap["TRAJ"].astype("int32")

        return df_soap