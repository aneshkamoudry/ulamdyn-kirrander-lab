__author__ = 'Max Pinheiro Jr <maxjr82@gmail.com>'
__date__   = 'Mar 14, 2021'

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

from DataLoader import *
from Descriptors import *
from DataWriter import *
from UnsupModels import *

def aggregate_data(df):
    cols_to_skip = ['time', 'State', 'TRAJ']
    col_names = df.columns.values.tolist()
    cols_to_skip = list(set(col_names).intersection(set(cols_to_skip)))
    vars_to_aggregate = {k: ['median', 'mean', 'std']
                         for k in df.drop(cols_to_skip, axis=1).columns.values}
    df_stats = df.groupby(['time'], as_index=False).agg(vars_to_aggregate)
    df_stats.columns = ['_'.join(col).strip() for col in df_stats.columns.values]
    df_stats.columns = [col.rstrip('_') for col in df_stats.columns.values]
    return df_stats

def calc_avg_occupations(df):
    # Compute the average occupations of the trajectories for each state state
    # STEP 1 - count the number of trajectories occupying a given state at each time
    x = df.groupby(['time','State']).count().reset_index()[['time', 'State', 'TRAJ']]
    # STEP 2 - divide the number of trajectories in a given state by the total number
    #          of successful trajectories
    x['Occ'] = x['TRAJ']/len(df['TRAJ'].unique())
    # STEP 3 - create num_states new columns with the respective occupations, and fill
    # missing values with zeros
    df_occ = x.pivot_table(values='Occ', index='time', 
                           columns='State', fill_value=0).reset_index()
    # STEP 4: rename columns (state value -> 'Occ + state value')
    df_occ.drop(['time'], axis = 1, inplace = True)
    df_occ.columns = ["Occ" + str(i) for i in df_occ.columns]
    return df_occ

def create_stats(selected_data):

    if selected_data == 'all':
        print("Calculating statistics for the properties dataset...\n")
        gp = GetProperties()
        df = gp.energies()
        df = gp.oscillator_strength()
        df = gp.populations()

        time_vec = df['time'].values
        df_prop_stats = aggregate_data(df)

        df_occ = calc_avg_occupations(df)        
        # Finally: merge the two dataframes
        df_prop_stats = pd.concat((df_prop_stats, df_occ), axis=1)
        
        print("Calculating statistics for the R2 descriptor...\n")
        gc = GetCoords()
        gc.read_all_trajs()
        gc.align_geoms
        r2 = R2()
        df = r2.build_descriptor(gc.xyz, save_csv=False)
        try:
            df['time'] = time_vec
        except:
            print("The length of the R2 and properties datasets does not match.")
            print("Please check the number of lines in each csv file.")
            sys.exit()

        df_r2_stats = aggregate_data(df)

        print("Calculating statistics for the Z-Matrix...\n")
        zmt = ZMatrix()
        df = zmt.build_descriptor(gc.xyz, save_csv=False)
        try:
            df['time'] = time_vec
        except:
            print("The length of the Z-Matrix and properties datasets does not match.")
            print("Please check the number of lines in each csv file.")
            sys.exit()
        
        df_zmt_stats = aggregate_data(df)

        all_stats = {'properties': df_prop_stats,
                     'r2': df_r2_stats,
                     'zmatrix': df_zmt_stats}
        return all_stats

def build_descriptor(descriptor):
    gc = GetCoords()
    gc.read_all_trajs()
    gc.align_geoms
    all_aligned_geoms = gc.xyz.copy()
    df_xyz = gc.build_dataframe(save_csv=False)
    if descriptor == 'aXYZ':
        return df_xyz
    elif descriptor == 'R2':
        r2 = R2()
        df_r2 = r2.build_descriptor(all_aligned_geoms, save_csv=False)
        return df_r2    
    elif descriptor == 'Z-Matrix':
        zmt = ZMatrix()
        df_zmt = zmt.build_descriptor(all_aligned_geoms, save_csv=False)
        return df_zmt 
    else:
        print("-------------------------------------------------")
        print("ERROR: Descriptor not recognized/implemented!\n")
        print("Please enter a valid descriptor:")
        print("aXYZ, R2 or Z-Matrix.")
        print("-------------------------------------------------")
        sys.exit()

def save_data(data_to_save):
    if data_to_save == 'all':
        print("Saving the full XYZ coordinates dataframe...\n")
        gc = GetCoords()
        gc.read_all_trajs()
        gc.align_geoms
        df = gc.build_dataframe(save_csv=True)
        print("Saving the full properties dataframe...\n")
        gp = GetProperties()
        df = gp.energies()
        df = gp.oscillator_strength()
        df = gp.populations()
        gp.save_csv
        print("Saving the R2 descriptor dataframe...\n")
        r2 = R2()
        all_geoms = gc.xyz.copy()
        df = r2.build_descriptor(all_geoms, save_csv=True)
        print("Saving the Z-Matrix descriptor dataframe...\n")
        zmt = ZMatrix()
        df = zmt.build_descriptor(all_geoms, save_csv=True)

def save_xyz_hoppings(states_pair):
    # 1) Generate the dataframe with all properties
    gp = GetProperties()
    df_props = gp.energies()
    df_props = gp.oscillator_strength()
    df_props = gp.populations()
    # 2) Load the XYZ coordinates from all trajectories
    gc = GetCoords()
    gc.read_all_trajs()
    gc.align_geoms

    col_hopping = 'Hops_' + states_pair
    if col_hopping in df_props.columns.tolist():
        indices = df_props[df_props[col_hopping] == 1].index.tolist()
        df_props = df_props[df_props[col_hopping] == 1].reset_index(drop=True)
        df_props = df_props.round(4)
        hopping_geoms = gc.xyz[indices].copy()
        atom_labels = gc.labels.reshape(-1,1)
        add_property = [states_pair.replace('S','DE')]
        out_name = 'Geoms_Hopping_' + states_pair + '.xyz'
        geoms = Geometries()
        geoms.save_xyz(atom_labels, hopping_geoms, df_props, add_property, out_name)
    else:
        print("There is no hopping for the selected pair of states.")    


if __name__ == '__main__':
        
    # Define command-line arguments. The commands can be read from a config.txt file.
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--create_stats", required=False, type=str, default=None,  
                        help="Generate a new dataset with the basic statistics \
                              (mean, median and std) over all the trajectories.")
    parser.add_argument("--save_dataset", required=False, type=str, default=None,  
                        help="Type of dataset (properties and/or descriptors) to\
                              build and save as csv file.")
    parser.add_argument("--save_xyz_hops", required=False, type=str, default=None,  
                        help="Write a single XYZ file with all hopping geometries\
                              for a given pair of states.")
    parser.add_argument("--dim_reduction", required=False, type=str, default=None,  
                        help="Select the model type for dimensionality reduction\
                              analysis.")
    parser.add_argument("--n_dim", required=False, type=int, default=2,  
                        help="Number of dimensions of the reduced data set.")
    parser.add_argument("--n_samples", required=False, type=int, default=None,  
                        help="Number of samples randomly selected from the data\
                              to use in the dimensionality reduction analysis.")                    
    parser.add_argument("--data_scaler", required=False, type=str, default=None,  
                        help="Method to rescale the data set before applying the\
                              unsupervised learning model.")                    
    parser.add_argument("--descriptor", required=False, type=str, default=None,  
                        help="Descriptor used to build the data set in which the\
                              unsupervised learning model will be applied.")
      
    # If no command-line arguments are present, config file is parsed
    config_file='config.txt'
    if len(sys.argv) == 1:
        if os.path.isfile(config_file):
            args = parser.parse_args(["@"+config_file])
        else:
            args = parser.parse_args(["--help"])
    else:
        args = parser.parse_args()

    if args.save_dataset is not None:
        save_data(args.save_dataset)

    if args.create_stats is not None:
        data_dict = create_stats(args.create_stats)
        for key in data_dict.keys():
            csv_name = 'stats_' + key + '.csv'
            df = data_dict[key]
            df.to_csv(csv_name, index=False, header=True, float_format="%.8f")

    if args.save_xyz_hops is not None:
        save_xyz_hoppings(args.save_xyz_hops)

    if args.dim_reduction is not None:
        # First step: create the dataset to apply the 
        #             dimensionality reduction model.
        df = build_descriptor(args.descriptor)
        # Second step: instanciate the dimensionality reduction class
        dimred = DimensionalityReduction(data=df, scaler=args.data_scaler)
        model = args.dim_reduction.lower().strip()
        if model == 'pca':
            df_reduced = dimred.pca(n_components=args.n_dim,calc_error=True)
            csv_name = 'pca_ndim' + str(args.n_dim) + '_'
            csv_name += args.descriptor.lower() + '.csv'
            df_reduced.to_csv(csv_name, index=True, header=True)
