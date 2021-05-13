## Author: Max Pinheiro Jr <maxjr82@gmail.com>
## Date: May 13, 2021
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


def get_properties_data():

    try:
        df = pd.read_csv('all_properties.csv')
        print("Loading properties data from existing csv...\n")
    except:
        gp = GetProperties()
        df = gp.energies()
        df = gp.oscillator_strength()
        df = gp.populations()

    return df    

def build_descriptor(descriptor, getcoords_obj):
    all_aligned_geoms = getcoords_obj.xyz
    # getcoords_obj.xyz is a variable of the class object
    # that stores all XYZ coordinates as a numpy array of 
    # dimension [n_geoms, n_atoms, 3]
    if descriptor == 'aXYZ':
        df_xyz = getcoords_obj.dataset
        df_xyz.to_csv(descriptor + '.csv', index=False)
        return df_xyz
    elif descriptor in ['R2', 'inv-R2', 'delta-R2', 'RE']:
        r2 = R2()
        df_r2 = r2.build_descriptor(all_aligned_geoms, descriptor)
        df_r2.to_csv(descriptor + '.csv', index=False)
        return df_r2
    elif descriptor in ['Zmat', 'delta-Zmat']:
        zmt = ZMatrix()
        dfs_dict = {'Zmat': zmt.build_descriptor(all_aligned_geoms),
                    'delta-Zmat': zmt.build_descriptor(all_aligned_geoms, delta=True)}
        df_zmt = dfs_dict[descriptor]
        df_zmt.to_csv(descriptor + '.csv', index=False)
        return df_zmt 
    else:
        print("-----------------------------------------------------")
        print("ERROR: \n")
        print("Descriptor not recognized or implemented!\n")
        print("Please select one of the available descriptors:")
        print("aXYZ, R2, inv-R2, delta-R2, RE, Zmat or delta-Zmat.")
        print("-----------------------------------------------------")
        sys.exit()

def run_dim_reduction(args):
    # Step 1: Load XYZ data from all trajectories and align coordinates
    gc = GetCoords()
    gc.read_all_trajs()
    gc.align_geoms
    gc.build_dataframe()

    # Step 2: create the dataset to apply the dimensionality reduction model.
    df = build_descriptor(args.descriptor, gc)

    # Step 3: build the dataset of properties that can be used for colormap.
    df_props = get_properties_data()

    try:
        df_props['RMSD'] = gc.rmsd
    except:
        print("--------------------------------------------------------")
        print("There is a mismatch in the length of coordinates ({})") 
        print("and properties ({}) data sets.".format(len(gc.rmsd),
                                                     df_props.shape[0]))
        print("The RMSD can not be added to the properties data set.")
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
