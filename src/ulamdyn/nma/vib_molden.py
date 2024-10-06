# author: Felix Plasser and Max Pinheiro Jr <maxjr82@gmail.com>
#   Date: June 2, 2023
#  usage: Package for reading a molden input vibration file.
#         This input is used for normal mode analysis by nma.py.
#         Also a multiple xyz vibration file compatible with Jmol can be
#         printed out.

import os

import numpy as np

from ulamdyn.nma import _file_handler as fh
from ulamdyn.nx_utils import BOHR_TO_ANG


def convert_molden2xyz(path=".", name="molden.input"):
    """
    Sub to convert a molden file to an xyz file.
    """
    vm2x = VibMolden()
    vm2x.read_molden_file(os.path.join(path, name))
    vm2x.print_xyz_file(os.path.join(path, name.rpartition(".")[0] + ".xyz"))


class VibMolden:
    """
    Class used to read frequencies and coordinates from a molden file.
    """

    def __str__(self) -> str:
        """Provide a string representation of the class.

        :return: Short description of the class functionality.
        :rtype: str
        """
        return "Read frequencies and coordinates from a molden file."

    def __init__(self) -> None:
        self.atoms = []
        self.freqs = []
        self.vibs = []

    def read_molden_file(self, file_path: str):
        molden_file = open(file_path, "r")
        Atoms = False
        FREQ = False
        FRNORMCOORD = False

        actvib = -1
        for line in molden_file:
            # what section are we in
            if "[" in line or "--" in line:
                Atoms = False
                FREQ = False
                FRNORMCOORD = False

            if "[Atoms]" in line:
                Atoms = True
            elif "[FREQ]" in line:
                FREQ = True
            elif "[FR-NORM-COORD]" in line:
                FRNORMCOORD = True
            # extract the information in that section
            elif Atoms:
                self.atoms += [
                    atom(
                        fh.line_to_words(line)[0],
                        fh.line_to_words(line)[3:6],
                    )
                ]
            elif FREQ:
                self.freqs += [eval(line)]
            elif FRNORMCOORD:
                if "vibration" in line:
                    actvib += 1
                    self.vibs += [vibration(self.freqs[actvib])]
                else:
                    self.vibs[actvib].add_vector(fh.line_to_words(line)[0:3])

    def get_vib_matrix(self):
        """
        Return a normalised matrix that contains the coordinates for all the
        vibrations. The modes are in the lines of this matrix!
        """
        m_vib = np.array([vib.ret_joined_vector() for vib in self.vibs])
        return m_vib

    def get_freqs(self):
        """
        Return the frequencies.
        """
        return self.freqs

    def ret_eff_masses(self, mol_calc, mass_wt_pw=1):
        """
        Returns a list with the effective masses for all the modes.
        <mol_calc> is a struc_linalg.mol_calc instance with the mass
        information.
        """
        return [
            vib.ret_eff_mass(mol_calc, mass_wt_pw=mass_wt_pw)
            for vib in self.vibs
        ]

    def get_vib_data(self):
        """Output the vibrational data for the reference structure."""
        nr_list = []
        freq_list = []  # frequency
        T_list = []  # period
        for nr, freq in enumerate(self.freqs):
            try:
                T = 1 / (freq * 2.9979e-5)
            except ZeroDivisionError:
                T = 0

            nr_list += ["NM" + str(nr + 1)]
            freq_list += [str(freq)[:6]]
            T_list += [str(T)[:6]]

        return [nr_list, freq_list, T_list]

    def print_xyz_file(self, file_path):
        """
        Creates an xyz file with the vibrational information.
        """
        out_str = ""
        for vib in self.vibs:
            out_str += str(len(self.atoms)) + "\n"
            out_str += "Frequency " + str(vib.frequency) + "\n"

            tmaker = fh.table_maker([5] + 6 * [20])
            for at_ind, atom in enumerate(self.atoms):
                tmaker.write_line(
                    [" " + atom.name] + atom.pos + vib.vector_list[at_ind]
                )
            out_str += tmaker.return_table()

        xyzfile = open(file_path, "w")
        xyzfile.write(out_str)
        xyzfile.close()


class atom:
    def __init__(
        self, name, pos, length_factor=BOHR_TO_ANG
    ):  # units are changed from Bohr into Angstrom
        "Position in A"
        self.name = name.capitalize()
        self.pos = [eval(coor) * length_factor for coor in pos]


class vibration:
    def __init__(self, frequency):
        self.vector_list = []
        self.frequency = frequency

    def add_vector(
        self, vector, length_factor=1.0
    ):  # in this case the units are not changed to keep orthonormality
        self.vector_list += [[eval(coor) * length_factor for coor in vector]]

    def ret_joined_vector(self):
        """
        Returns a joined 3N-dimensional numpy vector for the vibration.
        """
        t_list = []
        for vector in self.vector_list:
            t_list += vector
        t_list = np.array(t_list)
        return t_list

    def ret_eff_mass(self, mol_calc, mass_wt_pw=1):
        """
        Return the effective mass of the vibration according to
        structure <struc>.
        """
        M = mol_calc.ret_mass_matrix(power=mass_wt_pw)

        vec = self.ret_joined_vector()
        return np.dot(np.dot(vec, M), vec) / np.dot(vec, vec)


def make_molden_file(
    struc, freqs, vibs, out_file, title="Essential dynamics", num_at=None
):
    """
    Subroutine for making a molden file.
    """
    if num_at is None:
        num_at = struc.ret_num_at()

    out_str = "[Molden Format]\n[Title]\n" + title + "\n"

    if struc is not None:
        out_str += "[Atoms] AU\n"
        out_str += ret_Atoms_table(struc)

    # wavenumbers
    out_str += "[FREQ]\n"
    for freq in freqs:
        out_str += " " + str(freq) + "\n"

    if struc is not None:
        out_str += "[FR-COORD]\n"
        out_str += ret_FRCOORD_table(struc)

    # normal modes
    out_str += "[FR-NORM-COORD]\n"
    for ind in range(len(vibs)):
        out_str += " vibration" + str(ind + 1).rjust(5) + "\n"
        for j in range(num_at):
            out_str += "% 14.8f % 14.8f % 14.8f\n" % (
                vibs[ind][3 * j],
                vibs[ind][3 * j + 1],
                vibs[ind][3 * j + 2],
            )

    w_file = open(out_file, "w")
    w_file.write(out_str)
    w_file.close()


def ret_Atoms_table(struc):
    """
    Create the atoms part in the molden file.
    """
    tblmaker = fh.table_maker([6, 4, 3, 21, 21, 21])
    for i in range(struc.ret_num_at()):
        atom = struc.mol.GetAtom(i + 1)
        tblmaker.write_line(
            [struc.ret_symbol(i + 1), i + 1, atom.GetAtomicNum()]
            + [atom.x() / BOHR_TO_ANG]
            + [atom.y() / BOHR_TO_ANG]
            + [atom.z() / BOHR_TO_ANG]
        )

    return tblmaker.return_table()


def ret_FRCOORD_table(struc):
    """
    Create the FRCOORD part in the molden file.
    """
    tblmaker = fh.table_maker([4, 21, 21, 21])
    for i in range(struc.ret_num_at()):
        atom = struc.mol.GetAtom(i + 1)
        # vec = atom.GetVector()
        tblmaker.write_line(
            [struc.ret_symbol(i + 1)]
            + [atom.x() / BOHR_TO_ANG]
            + [atom.y() / BOHR_TO_ANG]
            + [atom.z() / BOHR_TO_ANG]
        )

    return tblmaker.return_table()
