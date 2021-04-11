## Author: Max Pinheiro Jr <maxjr82@gmail.com>
## Date: 03/14/2021

import os
import sys
import numpy as np

from DataLoader import *

class Geometries:

    def _info(self, dataframe, idx, props_to_print):
        
        comment_line = ""

        for p in props_to_print:
            val = dataframe[p][idx]
            string = p + " = " + str(val) + " | "
            comment_line += string
 
        return comment_line

    def save_xyz(self, labels, geoms_array, properties_data, 
                 add_properties=list(), output_name=None):
    
        n_atoms = len(labels)
        geoms_string = ""

        default_properties = ['TRAJ', 'time']
        props_to_print = default_properties + add_properties
        props_to_print = sorted(set(props_to_print), key=props_to_print.index)
        
        for n, xyz in enumerate(geoms_array):
            xyz = np.concatenate((labels, xyz), axis=1)
            xyz_str = [str(i).strip('[]') for i in xyz]
            comment_line = self._info(properties_data, n, props_to_print)
            geoms_string += str(n_atoms) + '\n' + comment_line + '\n'
            geoms_string += '\n'.join(xyz_str)
            geoms_string = geoms_string.replace("'", "")
            geoms_string += '\n'
                      
        if output_name == None:
            output_name = 'selected_geoms.xyz'
            
        with open(output_name, 'w') as out:
            out.write(geoms_string)        
