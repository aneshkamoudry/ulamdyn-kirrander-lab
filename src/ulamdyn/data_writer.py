# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: March 14, 2021

import numpy as np
import pandas as pd

from ulamdyn.base import BaseClass

__all__ = ["Geometries"]


class Geometries(BaseClass):
    """Handle and save XYZ coordinates for selected frames of
    MD trajectories."""

    def __str__(self) -> str:
        """Provide a description of the class functionality.

        :return: Human-readable string explaining the class functionality.
        :rtype: str
        """
        msg = "Module to export molecular geometries (or gradients) "
        msg += "in xyz format."
        return msg

    def __init__(self, atom_labels, properties_data=None, add_properties=[]):
        """Class initialization.

        :param atom_labels: array of atom labels used to write the XYZ
                            coordinates file.
        :type atom_labels: np.array
        :param properties_data: dataframe containing the property values of
                                the selected geometries, defaults to None.
        :type properties_data: pandas.DataFrame
        :param add_properties: list of properties to be added in the comment
                               line of the XYZ file.
                               Defaults to ["TRAJ", "time"]
        :type add_properties: list
        """
        self.labels = atom_labels.reshape(-1, 1)
        self.props_name = []
        self.props_data = properties_data
        if isinstance(self.props_data, pd.DataFrame):
            self.props_name = list(
                set(["TRAJ", "time"]).intersection(
                    properties_data.columns.tolist()
                )
            )
            self.props_data = self.props_data.round(4)

            if len(add_properties) != 0:
                self.props_name += add_properties
                self.props_name = sorted(
                    set(self.props_name), key=self.props_name.index
                )

    def _info(self, idx, props_to_print) -> str:
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
            val = self.props_data[p][idx]
            separator = " | " if p != props_to_print[-1] else ""
            string = p + " = " + str(val) + units + separator

            comment_line += string

        return comment_line

    def save_xyz(
        self, geoms_array: np.ndarray, out_name: str = "selected_geoms.xyz"
    ) -> None:
        """Save an XYZ file for a set of selected molecular geometries.

        .. note:: The comment line of the XYZ file will contain a list of
                  property values for each molecule.

        :param geoms_array: a 3D array containing the list of XYZ matrices.
        :type geoms_array: numpy.ndarray
        :param out_name: name of the XYZ file containing all the selected
                         geometries, defaults to selected_geoms.xyz.
        :type out_name: str
        """
        n_atoms = len(self.labels)
        geoms_string = ""
        comment_line = ""
        mask = "{:<6s} {:12.8f} {:12.8f} {:12.8f} \n"

        for n, xyz in enumerate(geoms_array):
            if len(self.props_name) > 0:
                comment_line = self._info(n, self.props_name)

            geoms_string += str(n_atoms) + "\n" + comment_line + "\n"
            for label, atom_coords in zip(self.labels, xyz):
                geoms_string += mask.format(label[0], *atom_coords)

        with open(out_name, "w") as out:
            out.write(geoms_string)
