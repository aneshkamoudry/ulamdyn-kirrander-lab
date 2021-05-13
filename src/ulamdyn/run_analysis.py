## Author: Max Pinheiro Jr <maxjr82@gmail.com>
## Date: March 10, 2021
from __future__ import (absolute_import, division, print_function,
                        unicode_literals, with_statement)

import os
import sys
import time
import argparse
import numpy as np

try:
    import modin.pandas as pd
    import ray
    ray.init()
except:
    import pandas as pd

from ulamdyn.data_loader import *
from ulamdyn.data_writer import *
from ulamdyn.descriptors import *
from ulamdyn.statistics import *
from ulamdyn.unsup_models import *
from ulamdyn.interface import *

def _check_geom_file():
    if not os.path.isfile('geom.xyz'):
        print("\n--------------------------------------------------------")
        print("ERROR:                                             \n")
        print("The geom.xyz file was not found.")
        print("Please provide a reference geometry (geom.xyz) in the ")
        print("working directory.")
        print("Aborting execution.")
        print("--------------------------------------------------------")
        sys.exit()
    
def save_data(data_to_save):
    if data_to_save == 'all':
        print("Saving the full XYZ coordinates dataframe...\n")
        gc = GetCoords()
        gc.read_all_trajs()
        gc.align_geoms
        gc.save_csv
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

    elif data_to_save == 'gradients':
        print("Saving the XYZ gradients for each available state as dataframes...\n")
        gg = GetGradients()
        gg.read_all_trajs()
        gg.build_dataframe(save_csv=True)

def save_xyz_hoppings(states_pair):
    # 1) Load the XYZ coordinates from all trajectories
    gc = GetCoords()
    gc.read_all_trajs()
    gc.align_geoms

    # 2) Generate the dataframe with all properties
    gp = GetProperties()
    df_props = gp.energies()
    df_props = gp.oscillator_strength()
    df_props = gp.populations()
    try:
        df_props['RMSD'] = gc.rmsd
    except:
        print("--------------------------------------------------------")
        print("There is a mismatch in the length of coordinates ({})") 
        print("and properties ({}) data sets.".format(len(gc.rmsd),
                                                     df_props.shape[0]))
        print("The RMSD will not be added to the properties data set.")
        print("--------------------------------------------------------")

    col_hopping = 'Hops_' + states_pair
    if col_hopping in df_props.columns.tolist():
        indices = df_props[df_props[col_hopping] == 1].index.tolist()
        df_props = df_props[df_props[col_hopping] == 1].reset_index(drop=True)
        df_props = df_props.round(4)
        hopping_geoms = gc.xyz[indices].copy()
        atom_labels = gc.labels.reshape(-1,1)
        add_property = [states_pair.replace('S','DE')]
        if 'RMSD' in df_props.columns:
            add_property.append('RMSD')
        out_name = 'Geoms_Hopping_' + states_pair + '.xyz'
        geoms = Geometries(atom_labels, add_property)
        geoms.save_xyz(hopping_geoms, df_props, out_name)
    else:
        print("-----------------------------------------------------")
        print("There is no hopping for the selected pair of states.")
        print("-----------------------------------------------------")

def run_bootstrap(args):

    print("Loading the all properties data...\n")
    try:
        df = pd.read_csv('all_properties.csv')
    except:
        gp = GetProperties()
        df = gp.energies()
        df = gp.oscillator_strength()
        df = gp.populations()

    print("Running the bootstrap algorithm...\n")
    
    bootstrap_options = args.bootstrap.split(",")

    save_csv = True if 'save' in bootstrap_options else False

    # Setting some default values for the bootstrap statistics
    n_samples = len(df['TRAJ'].unique())
    n_repeats = 1000
    ci_level = 95

    if len(bootstrap_options) >= 3:
        n_samples, n_repeats = list(map(int, bootstrap_options[:2]))
        # Check if the CI level is provided as input.
        # It must be the third option in the list.
        try:
            ci_level = int(bootstrap_options[2])
        except ValueError:
            pass

    elif len(bootstrap_options) == 2:
        if save_csv:
            n_repeats = int(bootstrap_options[0])
        else:
            n_repeats, ci_level = list(map(int, bootstrap_options[:1]))

    elif len(bootstrap_options) == 1:
        if not save_csv:
           n_repeats = int(bootstrap_options)
        
    print("The following parameters will be used in the bootstrap:\n")
    print("   number of trajectories = {:<20}".format(n_samples))
    print("        number of repeats = {:<20}".format(n_repeats))
    print("      confidence interval = {:<20}%".format(ci_level))
    print(" ")

    df_bootstrap = bootstrap(df,n_samples=n_samples,
                             n_repeats=n_repeats,
                             save_csv=save_csv)

    print("Creating statistics for the bootstrapped data...\n")
    df_stats = create_bootstrap_stats(df_bootstrap, ci_level)
    df_stats.to_csv("stats_bootstrap.csv", index=False)


if __name__ == '__main__':
        
    # Define command-line arguments. The commands can be read from a config.txt file.
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--save_dataset", required=False, type=str, metavar='', default=None,  
                        help="Type of data set (properties and/or descriptors) to build \
                              and save as csv file.")
    parser.add_argument("--save_xyz_hops", required=False, type=str, metavar='', default=None,  
                        help="Write a single XYZ file with all hopping geometries for a \
                              given pair of states.")
    parser.add_argument("--create_stats", required=False, type=str, metavar='', default=None,  
                        help="Generate a data set with basic statistics (mean, median \
                              and std) for all the trajectories.")
    parser.add_argument("--bootstrap", required=False, type=str, metavar='', default=None,  
                        help="Run the bootstrap algorithm for the properties data set \
                              and save a new data with basic statistics (mean, median \
                              and std) and confidence intervals.")                                                    
    parser.add_argument("--descriptor", required=False, type=str, metavar='', default='inv-R2',
                        help="Type of molecular descriptor used in the unsupervised\
                              learning analysis.")
    parser.add_argument("--dim_reduction", required=False, type=str, metavar='', default=None,  
                        help="Select a model for the dimensionality reduction analysis:\
                              PCA, KPCA, Isomap or tSNE.")
    parser.add_argument("--n_dim", required=False, type=int, metavar='', default=2,  
                        help="Number of dimensions of the reduced data set.")
    parser.add_argument("--n_samples", required=False, type=int, metavar='', default=None,  
                        help="Number of samples randomly selected from the data set.")
    parser.add_argument("--kernel", required=False, type=str, metavar='', default='rbf',  
                        help="Kernel function used for KPCA method.")
    parser.add_argument("--perplexity", required=False, type=float, metavar='', default=40,  
                        help="Perplexity parameters used in the t-SNE algorithm.")                    
    parser.add_argument("--data_scaler", required=False, type=str, metavar='', default='',  
                        help="Method to rescale the data set before applying the \
                              unsupervised learning model.")
    parser.add_argument("--n_cpus", required=False, type=int, metavar='', default=-1,  
                        help="Number of CPUs allocated for the parallelization of \
                              the unsupervised learning tasks. If -1 all available \
                              processors will be used.")
      
    # If no command-line arguments are present, config file is parsed
    config_file='config.txt'
    if len(sys.argv) == 1:
        if os.path.isfile(config_file):
            args = parser.parse_args(["@"+config_file])
        else:
            args = parser.parse_args(["--help"])
    else:
        args = parser.parse_args()

    _check_geom_file()

    start = time.time()

    if args.save_dataset is not None:
        save_data(args.save_dataset)

    if args.save_xyz_hops is not None:
        save_xyz_hoppings(args.save_xyz_hops)

    if args.create_stats is not None:
        data_dict = create_stats(args.create_stats)
        for key in data_dict.keys():
            csv_name = 'stats_' + key + '.csv'
            df = data_dict[key]
            df.to_csv(csv_name, index=False, header=True, float_format="%.8f")

    if args.bootstrap is not None:
        run_bootstrap(args)

    if args.dim_reduction is not None:
        run_dim_reduction(args)    

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    print("---------------------------------------------------")
    print("Total execution time - {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    print("---------------------------------------------------")
    print(" ")

    print("Finished!")    
