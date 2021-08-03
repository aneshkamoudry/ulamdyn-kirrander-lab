# __author__ = 'Max Pinheiro Jr <maxjr82@gmail.com>'
# __date__ = '03/14/2021'
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
    with_statement,
)

import os, sys
import numpy as np
from ulamdyn.data_loader import *

__all__ = ["Geometries"]


class Geometries:
    """Handle and save XYZ coordinates for selected frames of MD trajectories."""

    def __repr__(self) -> str:
        return "Module to export molecular geometries in xyz format."

    def __init__(self, atom_labels, add_properties=list()):
        """Class initialization.

        :param atom_labels: array of atom labels used to write the XYZ coordinates file.
        :type atom_labels: np.array
        :param add_properties: list of properties to be added in the comment line of the XYZ file, defaults to ["TRAJ", "time"]
        :type add_properties: list()
        """
        self.labels = atom_labels
        self.properties = ["TRAJ", "time"]

        if len(add_properties) != 0:
            self.properties += add_properties
            self.properties = sorted(set(self.properties), key=self.properties.index)

    def _info(self, dataframe, idx, props_to_print):

        comment_line = ""

        for p in props_to_print:
            units = ""
            if "DE" in p:
                states = list(p.replace("DE", ""))
                states = "".join(sorted(states, key=int, reverse=True))
                p = "DE" + states
                units = " eV"
            if "time" in p:
                units = " fs"
            if "RMSD" in p:
                units = " ang"
            val = dataframe[p][idx]
            separator = " | " if p != props_to_print[-1] else ""
            string = p + " = " + str(val) + units + separator

            comment_line += string

        return comment_line

    def save_xyz(
        self, geoms_array, properties_data=None, out_name="selected_geoms.xyz"
    ):
        """Save an XYZ file for a set of selected molecular geometries.

        .. note:: The comment line of the XYZ file will contain a list of property values
                  for each molecule.

        :param geoms_array: a 3D array containing the list of XYZ matrices.
        :type geoms_array: numpy.ndarray
        :param properties_data: dataframe containing the property values of the selected geometries, defaults to None.
        :type properties_data: pandas.DataFrame
        :param out_name: name of the XYZ file containing all the selected geometries, defaults to selected_geoms.xyz.
        :type out_name: str
        """
        n_atoms = len(self.labels)
        geoms_string = ""
        comment_line = ""

        for n, xyz in enumerate(geoms_array):
            xyz = np.round(xyz, 8)
            xyz = np.concatenate((self.labels, xyz), axis=1)
            xyz_str = [str(i).strip("[]") for i in xyz]

            if properties_data is not None:
                comment_line = self._info(properties_data, n, self.properties)

            geoms_string += str(n_atoms) + "\n" + comment_line + "\n"
            geoms_string += "\n".join(xyz_str)
            geoms_string = geoms_string.replace("'", "")
            geoms_string += "\n"

        with open(out_name, "w") as out:
            out.write(geoms_string)
