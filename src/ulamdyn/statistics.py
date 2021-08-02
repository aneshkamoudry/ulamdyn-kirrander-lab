## Author: Max Pinheiro Jr <maxjr82@gmail.com>
## Date: April 25, 2021
from __future__ import (absolute_import, division, print_function,
                        unicode_literals, with_statement)

import os
import sys
import time
import numpy as np

try:
    import modin.pandas as pd
    import ray

    ray.shutdown()
    ray.init()
except:
    import pandas as pd

from ulamdyn.data_loader import *
from ulamdyn.descriptors import *

def aggregate_data(data,vars_to_group=['time']):
    skip_cols = ['time', 'State', 'TRAJ']
    if vars_to_group != ['time']:
        skip_cols = vars_to_group
    col_names = data.columns.values.tolist()
    skip_cols = list(set(col_names).intersection(set(skip_cols)))
    vars_to_aggregate = {k: ['median', 'mean', 'std']
                         for k in data.drop(skip_cols, axis=1).columns.values}
    df_stats = data.groupby(vars_to_group, as_index=False).agg(vars_to_aggregate)
    df_stats.columns = ['_'.join(col).strip() for col in df_stats.columns.values]
    df_stats.columns = [col.rstrip('_') for col in df_stats.columns.values]
    return df_stats

def aggregate_by_time(df):
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

def _add_column(dataframe, col_name, array):
    try:
        dataframe[col_name] = array
    except:
        print("-----------------------------------------------------------------")
        print("ERROR:                                          \n")
        print("The size of the data set (n_rows = {}) and the inserted\n{} property \
               (n_rows = {}) does not match.".format(dataframe.shape[0], col_name, 
               array.shape[0]))
        print("Please check the number of lines in the related csv files.")
        print("-----------------------------------------------------------------")
        sys.exit()
    return dataframe    

def stats_hopping(dataframe):
    hopping_cols = list(filter(lambda k: 'Hops' in k, dataframe.columns.tolist()))
    
    if len(hopping_cols) == 0:
        print("---------------------------------------------")
        print("There is no hopping points in the data set.")
        print("The statistics can not be computed.")
        print("---------------------------------------------\n")
        return
    else:
        grouped = dataframe.groupby(['TRAJ'])[hopping_cols]
        df_stats = grouped.sum()
        
        for k in hopping_cols:
            stats = ['min', 'max']
            df1 = dataframe[dataframe[k] == 1].groupby(['TRAJ'])[['time']].agg(stats)
            new_col_labels = df1.columns.map(''.join).str.strip()
            s = k.replace('Hops_','')
            df1.columns = [s + '_' + i.replace('time','t') for i in new_col_labels]
            
            ordered_states = tuple(s.replace('S',''))
            ordered_states = ''.join(sorted(ordered_states, key=int, reverse=True))
            DE_col = 'DE' + ordered_states
            df2 = dataframe[dataframe[k] == 1].groupby(['TRAJ'])[[DE_col]].agg(stats)
            df2.columns = df2.columns.map('_'.join).str.strip('_')
            df_stats = pd.concat([df_stats, df1, df2], axis=1)
            
        df_stats = df_stats.reset_index()    
            
    return df_stats

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
        df_prop_stats = pd.concat((df_prop_stats, df_occ), axis=1)

        print("Calculating statistics for the hopping points...\n")

        df_hopping_stats = stats_hopping(df)
        
        print("Calculating statistics for the R2 descriptor...\n")
        gc = GetCoords()
        gc.read_all_trajs()
        gc.align_geoms
        r2 = R2()
        df = r2.build_descriptor(gc.xyz, save_csv=False)
        df = _add_column(df,'time',time_vec)
        df = _add_column(df, 'RMSD', gc.rmsd)
        df_r2_stats = aggregate_data(df)

        print("Calculating statistics for the Z-Matrix...\n")
        zmt = ZMatrix()
        df = zmt.build_descriptor(gc.xyz, save_csv=False)
        df = _add_column(df,'time',time_vec)
        df_zmt_stats = aggregate_data(df)

        all_stats = {'properties': df_prop_stats,
                     'hoppings': df_hopping_stats,
                     'r2': df_r2_stats,
                     'zmatrix': df_zmt_stats}

        return all_stats

def calc_ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return np.nanpercentile(a, p, axis)

def bootstrap(dataframe, n_samples=None, n_repeats=1000, save_csv=False):
    if not n_samples:
        n_samples = dataframe.shape[0]

    trajs = dataframe['TRAJ'].unique()
    trajs_sample = np.random.choice(trajs, size=n_samples, replace=True)

    df_bootstrap = pd.DataFrame()

    for i in range(n_repeats):
        selected_trajs = np.random.choice(trajs_sample, size=n_samples, replace=True)
        df = dataframe[dataframe['TRAJ'].isin(selected_trajs)]
        idx = [y for x in selected_trajs for y in df.index[df['TRAJ'].values == x]]
        df = df.loc[idx].reset_index(drop=True)
        df = df.groupby(['time'], as_index=False).mean()
        df_bootstrap = df_bootstrap.append(df)

    if save_csv:
        df_bootstrap.to_csv("bootstrapped_data.csv", index=False)

    return df_bootstrap

def create_bootstrap_stats(boot_data, ci_level=95):
    skip_cols = [i for i in boot_data.columns if 'Hops' in i]
    skip_cols += ['time', 'TRAJ', 'State']

    ci_dataframes = []

    for c in boot_data.columns.tolist():
        if c not in skip_cols:
            col_names = [c + '_ci_' + i for i in ['low', 'high']]
            grouped = boot_data.groupby(['time'])[c]
            df = pd.DataFrame(grouped.apply(calc_ci,which=ci_level).reset_index()[c].to_list(), 
                              columns=col_names)
            ci_dataframes.append(df)

    df_cis = pd.concat(ci_dataframes, axis=1)
    skip_cols.remove('time')
    boot_data = boot_data.drop(skip_cols, axis=1)
    df_stats_boot = boot_data.groupby('time').agg(['mean', 'median', 'std'])
    df_stats_boot.columns = df_stats_boot.columns.map('_'.join).str.strip()
    df_stats_boot = df_stats_boot.reset_index()
    df_stats_boot = pd.concat([df_stats_boot, df_cis], axis=1)

    old_cols_order = df_stats_boot.columns.tolist()
    new_cols_order = list()
    prefix = [c.split('_')[0] for c in old_cols_order]
    prefix = list(dict.fromkeys(prefix))
    for c in prefix:
        l = df_stats_boot.filter(regex=c,axis=1).columns.tolist()
        for i in l:
            new_cols_order.append(i)

    df_stats_boot = df_stats_boot[new_cols_order]

    return df_stats_boot
