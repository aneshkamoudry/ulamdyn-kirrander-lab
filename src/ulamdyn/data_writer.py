#__author__ = 'Max Pinheiro Jr <maxjr82@gmail.com>'
#__date__ = '03/14/2021'
from __future__ import (absolute_import, division, print_function,
                        unicode_literals, with_statement)

import os
import sys
import numpy as np

from ulamdyn.data_loader import *

__all__ = ['Geometries']

class Geometries:
    """Class used to save XYZ coordinates for selected frames of MD trajectories.

    This class requires two positional arguments in its constructor:
    - atom labels (np.array)
    - list of property names (the default properties are trajectory ID and time)

    Data attributes:
    ----------------
       ``labels`` (np.array): array of atom labels used to write the XYZ coordinates file.
       ``properties`` (list): list of properties to be added in the comment line of the XYZ file.

    Methods:
    --------
       ``save_xyz``: reads a set of geometries (np.array) and corresponding properties data to 
                     save this information in a XYZ file.
    
    """

    def __init__(self, atom_labels, add_properties=list()):
        self.labels = atom_labels
        self.properties = ['TRAJ', 'time']

        if len(add_properties) != 0:
            self.properties += add_properties
            self.properties = sorted(set(self.properties), 
                                     key=self.properties.index)

    def _info(self, dataframe, idx, props_to_print):
        
        comment_line = ""

        for p in props_to_print:
            units = ''
            if 'DE' in p:
                states = list(p.replace('DE',''))
                states = ''.join(sorted(states, key=int, reverse=True))
                p = 'DE' + states
                units = ' eV'
            if 'time' in p:
                units = ' fs'
            if 'RMSD' in p:
                units = ' ang'
            val = dataframe[p][idx]
            separator = " | " if p != props_to_print[-1] else ""
            string = p + " = " + str(val) + units + separator

            comment_line += string
 
        return comment_line

    def save_xyz(self, geoms_array, properties_data, out_name=None):
        """Save an XYZ file for a set of selected molecular geometries.

        .. note:: The comment line of the XYZ file will contain a list of property values
                  for each molecule.

        Args:
           geoms_array (np.array): a 3D array containing the list of XYZ matrices.
           properties_data (pd.dataframe): pandas dataframe containing the property values 
                                           of the selected geometries.
           out_name (str): name of the XYZ file containing all the selected geometries.
                           The default is selected_geoms.xyz.
    
        Returns:
           None:

        """
    
        n_atoms = len(self.labels)
        geoms_string = ""
      
        for n, xyz in enumerate(geoms_array):
            xyz = np.round(xyz, 8)
            xyz = np.concatenate((self.labels, xyz), axis=1)
            xyz_str = [str(i).strip('[]') for i in xyz]
            comment_line = self._info(properties_data, n, self.properties)
            geoms_string += str(n_atoms) + '\n' + comment_line + '\n'
            geoms_string += '\n'.join(xyz_str)
            geoms_string = geoms_string.replace("'", "")
            geoms_string += '\n'
                      
        if out_name == None:
            out_name = 'selected_geoms.xyz'
            
        with open(out_name, 'w') as out:
            out.write(geoms_string)        
