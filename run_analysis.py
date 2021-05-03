__author__ = 'Max Pinheiro Jr <maxjr82@gmail.com>'
__date__   = 'Mar 14, 2021'

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

from .DataLoader import *
from .Descriptors import *
from .DataWriter import *
from .Statistics import *
from .UnsupModels import *

def _check_geom_file():
    if not os.path.isfile('geom.xyz'):
        print("\n--------------------------------------------------------")
        print("ERROR:                                             \n")
        print("geom.xyz file not found.")
        print("Please provide a reference geometry [geom.xyz] in the ")
        print("working directory.")
        print("Aborting execution.")
        print("--------------------------------------------------------")
        sys.exit()
    
def build_descriptor(descriptor, getcoords_obj):
    all_aligned_geoms = getcoords_obj.xyz
    # getcoords_obj.xyz is a variable of the class object
    # that stores all XYZ coordinates as a numpy array of 
    # dimension [n_geoms, n_atoms, 3]
    if descriptor == 'aXYZ':
        df_xyz = getcoords_obj.dataset
        df_xyz.to_csv(descriptor + '.csv', index=False)
        return df_xyz
    elif descriptor == 'R2' or descriptor == 'inv-R2' or descriptor == 'delta-R2':
        r2 = R2()
        dfs_dict = {'R2': r2.build_descriptor(all_aligned_geoms),
                    'inv-R2': 1/r2.build_descriptor(all_aligned_geoms),
                    'delta-R2': r2.build_descriptor(all_aligned_geoms, delta=True)}
        df_r2 = dfs_dict[descriptor]
        df_r2.to_csv(descriptor + '.csv', index=False)
        return df_r2
    elif descriptor == 'Zmat' or descriptor == 'delta-Zmat':
        zmt = ZMatrix()
        dfs_dict = {'Zmat': zmt.build_descriptor(all_aligned_geoms),
                    'delta-Zmat': zmt.build_descriptor(all_aligned_geoms, delta=True)}
        df_zmt = dfs_dict[descriptor]
        df_zmt.to_csv(descriptor + '.csv', index=False)
        return df_zmt 
    else:
        print("---------------------------------------------------")
        print("ERROR: \n")
        print("Descriptor not recognized or implemented!\n")
        print("Please select one of the available descriptors:")
        print("aXYZ, R2, inv-R2, delta-R2, Zmat or delta-Zmat.")
        print("---------------------------------------------------")
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
        geoms = Geometries()
        geoms.save_xyz(atom_labels, hopping_geoms, df_props, add_property, out_name)
    else:
        print("-----------------------------------------------------")
        print("There is no hopping for the selected pair of states.")
        print("-----------------------------------------------------")

def run_dim_reduction(args):
    # Step 1: Load XYZ data from all trajectories and align coordinates
    gc = GetCoords()
    gc.read_all_trajs()
    gc.align_geoms
    gc.build_dataframe()

    # Step 2: create the dataset to apply the dimensionality reduction model.
    df = build_descriptor(args.descriptor, gc)

    # Step 3: build the dataset of properties that can be used for colormap.
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

    # Step 4: instanciate the dimensionality reduction class
    dimred = DimensionalityReduction(data=df, n_samples=args.n_samples,
                                     scaler=args.data_scaler, 
                                     n_cpus=args.n_cpus)
    model = args.dim_reduction.lower().strip()

    # Step 5: check for the available models and run the calculation
    if model == 'pca':
        df_reduced = dimred.pca(n_components=args.n_dim,calc_error=True)
    elif model == 'kpca':
        df_reduced = dimred.kpca(n_components=args.n_dim,kernel=args.kernel)
    elif model == 'isomap':
        df_reduced = dimred.isomap(n_components=args.n_dim,calc_error=True)
    elif model == 'tsne':
        df_reduced = dimred.tsne(n_components=args.n_dim,
                                 perplexity=args.perplexity)
    else:
        print("--------------------------------------------------------")
        print("ERROR:                                             \n")
        print("Model type not recognized or not implemented!")
        print("Please select one of the available methods:")
        print("PCA, KPCA, Isomap or t-SNE.")
        print("--------------------------------------------------------")
        sys.exit()    
    
    # Step 5: Save a csv file with the merged datasets (reduced + properties) 
    print("Saving the merged (reduced + properties) data sets...\n")
    df_reduced = df_reduced.merge(df_props, left_index=True, 
                                      right_index=True, how='left')
    csv_name = model + '_ndim' + str(args.n_dim) + '_'
    csv_name += args.descriptor.lower() + '.csv'
    df_reduced.to_csv(csv_name, header=True, index=True, index_label='index')

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
                              PCA, KPCA or Isomap.")
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