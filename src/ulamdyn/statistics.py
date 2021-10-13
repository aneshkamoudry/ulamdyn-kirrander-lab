"""Module to perform statistical analysis of the MD datasets."""
# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: April 25, 2021
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
    with_statement,
)

import os
import sys
import numpy as np

try:
    import modin.pandas as pd

except ModuleNotFoundError:
    import pandas as pd

    print("Modin package is not available.")
    print("The standard pandas library will be used.")

from ulamdyn.data_loader import GetCoords, GetProperties
from ulamdyn.descriptors import R2, ZMatrix
from ulamdyn.kinetics import KineticEnergy, VibrationalSpectra


def aggregate_data(data, vars_to_group=["time"]):
    """Calculate the basic statistical descriptors for a given dataset.

    The statistical quantities calculated by the function are *mean* and *median*
    to describe the central tendency, and *standard deviation* to measure the
    variability or dispersion of the data. So each feature of the original input
    dataset will be unfolded into three new columns identified with the suffixes
    '_median', '_mean' and '_std'.

    :param data: input dataset containing the information extracted from all MD
                 trajectories.
    :type data: pandas.DataFrame
    :param vars_to_group: set of variables used by the function to group the data,
                          defaults to ["time"]
    :type vars_to_group: list, optional
    :return: [description]
    :rtype: pandas.DataFrame | modin.pandas.dataframe.DataFrame
    """
    skip_cols = ["time", "State", "TRAJ"]
    if vars_to_group != ["time"]:
        skip_cols = vars_to_group
    col_names = data.columns.values.tolist()
    skip_cols = list(set(col_names).intersection(set(skip_cols)))
    vars_to_aggregate = {
        k: ["median", "mean", "std"]
        for k in data.drop(skip_cols, axis=1).columns.values
    }
    df_stats = data.groupby(vars_to_group, as_index=False).agg(vars_to_aggregate)
    df_stats.columns = ["_".join(col).strip() for col in df_stats.columns.values]
    df_stats.columns = [col.rstrip("_") for col in df_stats.columns.values]
    return df_stats


def calc_avg_occupations(df):
    """Calculate the fraction of trajectories in each state as a function of time.

    .. note:: The fraction of trajectories (occupation) is an important quantity to
              assess the quality of the surface hopping (SH) simulations. It should be
              compared to the population of each state averaged over all trajectories,
              which is provided by the method :meth:`~ulamdyn.aggregate_data`. If the
              ensemble of SH trajectories is statistically converged, the occupation
              and the average population should match.

    :param df: properties dataset with information collected from the available
               MD trajectories
    :type df: pandas.DataFrame | modin.pandas.dataframe.DataFrame
    :return: a new dataset with the occupations calculated for each state
    :rtype: pandas.DataFrame | modin.pandas.dataframe.DataFrame
    """
    # Compute the average occupations of the trajectories for each state state
    # STEP 1 - count the number of trajectories occupying a given state at each time
    x = df.groupby(["time", "State"]).count().reset_index()[["time", "State", "TRAJ"]]
    # STEP 2 - divide the number of trajectories in a given state by the total number
    #          of successful trajectories
    x["Occ"] = x["TRAJ"] / len(df["TRAJ"].unique())
    # STEP 3 - create num_states new columns with the respective occupations, and fill
    # missing values with zeros
    df_occ = x.pivot_table(
        values="Occ", index="time", columns="State", fill_value=0
    ).reset_index()
    # STEP 4: rename columns (state value -> 'Occ + state value')
    df_occ.drop(["time"], axis=1, inplace=True)
    df_occ.columns = ["Occ" + str(i) for i in df_occ.columns]
    return df_occ


def _add_column(dataframe, col_name, array):
    try:
        dataframe[col_name] = array
    except ValueError:
        print("-----------------------------------------------------------------")
        print("ERROR:                                          \n")
        print(
            "The size of the data set (n_rows = {}) and the inserted\n{} property \
               (n_rows = {}) does not match.".format(
                dataframe.shape[0], col_name, array.shape[0]
            )
        )
        print("Please check the number of lines in the related csv files.")
        print("-----------------------------------------------------------------")
        sys.exit()
    return dataframe


def stats_hopping(dataframe):
    """Generate a statistical summary for the hopping points.

    :param dataframe: properties dataset with information collected from the available
                      MD trajectories
    :type dataframe: pandas.DataFrame | modin.pandas.dataframe.DataFrame
    :return: a new dataset containing the total number of hops per trajectory and
             the minimum and maximum time in which the hops between each pair of
             electronic states occur.
    :rtype: pandas.DataFrame | modin.pandas.dataframe.DataFrame
    """
    hopping_cols = list(filter(lambda k: "Hops" in k, dataframe.columns.tolist()))

    if len(hopping_cols) == 0:
        print("---------------------------------------------")
        print("There is no hopping points in the data set.")
        print("The statistics can not be computed.")
        print("---------------------------------------------\n")
        return
    else:
        grouped = dataframe.groupby(["TRAJ"])[hopping_cols]
        df_stats = grouped.sum()

        for k in hopping_cols:
            stats = ["min", "max"]
            df1 = dataframe[dataframe[k] == 1].groupby(["TRAJ"])[["time"]].agg(stats)
            new_col_labels = df1.columns.map("".join).str.strip()
            s = k.replace("Hops_", "")
            df1.columns = [s + "_" + i.replace("time", "t") for i in new_col_labels]

            ordered_states = tuple(s.replace("S", ""))
            ordered_states = "".join(sorted(ordered_states, key=int, reverse=True))
            DE_col = "DE" + ordered_states
            df2 = dataframe[dataframe[k] == 1].groupby(["TRAJ"])[[DE_col]].agg(stats)
            df2.columns = df2.columns.map("_".join).str.strip("_")
            df_stats = pd.concat([df_stats, df1, df2], axis=1)

        df_stats = df_stats.reset_index()

    return df_stats


def create_stats(selected_data, save_csv=False):
    """Create datasets with the statistical summary of the requested quantities.

    :param selected_data: type of data used as input to compute the descriptive statistical
                          properties using the :meth:`~ulamdyn.aggregate_data` function
    :type selected_data: str
    :param save_csv: if true exports the calculated statistics for each dataframe in a csv
                     format, defaults to false
    :type save_csv: bool, optional
    :return: one or several datasets containing the median, mean and standard deviation
             calculated for each feature of the input as a function of time; if the
             attribute is equal to 'all', the statistics will be calculated for the
             properties and descriptors (R2 and Z-Matrix) datasets.
    :rtype: dict(pandas.DataFrame) | dict(modin.pandas.dataframe.DataFrame)
    """
    all_stats = dict()

    if selected_data.lower() == "all":
        print("Calculating statistics for the properties dataset...\n")
        gp = GetProperties()
        df = gp.energies()
        df = gp.oscillator_strength()
        df = gp.mcscf_coefs()
        df = gp.populations()

        time_vec = df["time"].values
        df_prop_stats = aggregate_data(df)

        df_occ = calc_avg_occupations(df)
        df_prop_stats = pd.concat((df_prop_stats, df_occ), axis=1)
        all_stats["properties"] = df_prop_stats

        print("Calculating statistics for the hopping points...\n")
        df_hopping_stats = stats_hopping(df)
        all_stats["hoppings"] = df_hopping_stats

        print("Calculating statistics for the R2 descriptor...\n")
        gc = GetCoords()
        gc.read_all_trajs()
        gc.align_geoms
        r2 = R2()
        df = r2.build_descriptor(gc.xyz, save_csv=False)
        df = _add_column(df, "time", time_vec)
        df = _add_column(df, "RMSD", gc.rmsd)
        df_r2_stats = aggregate_data(df)
        all_stats["r2"] = df_r2_stats

        print("Calculating statistics for the Z-Matrix...\n")
        zmt = ZMatrix()
        df = zmt.build_descriptor(gc.xyz, save_csv=False)
        df = _add_column(df, "time", time_vec)
        df_zmt_stats = aggregate_data(df)
        all_stats["zmatrix"] = df_zmt_stats

    elif selected_data.lower() == "ekin":
        print("Calculating statistics for the kinetic energies dataset...\n")
        gp = GetProperties()
        df = gp.energies()
        time_vec = df["time"].values

        ke = KineticEnergy()
        df = ke.build_dataframe()
        df.insert(loc=0, column="time", value=time_vec)
        df_ekin_stats = aggregate_data(df)
        all_stats["ekinetics"] = df_ekin_stats

    elif selected_data.lower() == "vibspec":
        print("Calculating statistics for the vibrational spectra dataset...\n")
        gp = GetProperties()
        df = gp.energies()
        time_vec = df["time"].values

        vs = VibrationalSpectra()
        df = vs.build_dataframe()
        df.insert(loc=0, column="time", value=time_vec)
        df_spec_stats = aggregate_data(df)
        all_stats["vibspec"] = df_spec_stats

    else:
        print("--------------------------------------")
        print("ERROR:                              \n")
        print("Invalid option!                       ")
        print("The available options are all or ekin.")
        print("--------------------------------------")
        return

    if save_csv:
        for key, df in all_stats.items():
            csv_name = "stats_" + key + ".csv"
            df.to_csv(csv_name, index=False, header=True, float_format="%.8f")

    return all_stats


def _calc_ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return np.nanpercentile(a, p, axis)


def bootstrap(dataframe, n_samples=None, n_repeats=1000, save_csv=False):
    """Estimate the mean by resampling the data with replacement.

    :param dataframe: input dataset having the quantities extrated from all available
                      MD trajectories; the dataset must contain the 'TRAJ' and 'time'
                      columns
    :type dataframe: pandas.DataFrame | modin.pandas.dataframe.DataFrame
    :param n_samples: number of trajectories considered in each round of the resampling,
                      defaults to None
    :type n_samples: int, optional
    :param n_repeats: number of resampling to be performed, defaults to 1000
    :type n_repeats: int, optional
    :param save_csv: if true outputs the final bootstrapped data in a csv file,
                     defaults to False
    :type save_csv: bool, optional
    :return: dataset containing all the bootstrapped data with estimated mean
    :rtype: pandas.DataFrame | modin.pandas.dataframe.DataFrame
    """
    if not n_samples:
        n_samples = dataframe.shape[0]

    trajs = dataframe["TRAJ"].unique()
    trajs_sample = np.random.choice(trajs, size=n_samples, replace=True)

    df_bootstrap = pd.DataFrame()

    for i in range(n_repeats):
        selected_trajs = np.random.choice(trajs_sample, size=n_samples, replace=True)
        df = dataframe[dataframe["TRAJ"].isin(selected_trajs)]
        idx = [y for x in selected_trajs for y in df.index[df["TRAJ"].values == x]]
        df = df.loc[idx].reset_index(drop=True)
        df = df.groupby(["time"], as_index=False).mean()
        df_bootstrap = df_bootstrap.append(df)

    if save_csv:
        df_bootstrap.to_csv("bootstrapped_data.csv", index=False)

    return df_bootstrap


def create_bootstrap_stats(boot_data, ci_level=95):
    """Generate the descriptive statistics for the bootstrapped data within a given confidence interval.

    :param boot_data: bootstrapped dataframe generate by the :meth:`~ulamdyn.bootstrap` function
    :type boot_data: pandas.DataFrame | modin.pandas.dataframe.DataFrame
    :param ci_level: Size of the confidence interval to draw when aggregating the data (by time)
                     to estimate the statistical descriptors (median, mean, std), defaults to 95
    :type ci_level: int, optional
    :return: dataframe grouped by the simulation time with each feature of the original dataset
             unfolded into three columns representing the statistical descriptors and other two columns storing the values for the lowest and highest variability as defined by the
             confidence interval
    :rtype: pandas.DataFrame | modin.pandas.dataframe.DataFrame
    """
    skip_cols = [i for i in boot_data.columns if "Hops" in i]
    skip_cols += ["time", "TRAJ", "State"]

    ci_dataframes = []

    for c in boot_data.columns.tolist():
        if c not in skip_cols:
            col_names = [c + "_ci_" + i for i in ["low", "high"]]
            grouped = boot_data.groupby(["time"])[c]
            df = pd.DataFrame(
                grouped.apply(_calc_ci, which=ci_level).reset_index()[c].to_list(),
                columns=col_names,
            )
            ci_dataframes.append(df)

    df_cis = pd.concat(ci_dataframes, axis=1)
    skip_cols.remove("time")
    boot_data = boot_data.drop(skip_cols, axis=1)
    df_stats_boot = boot_data.groupby("time").agg(["mean", "median", "std"])
    df_stats_boot.columns = df_stats_boot.columns.map("_".join).str.strip()
    df_stats_boot = df_stats_boot.reset_index()
    df_stats_boot = pd.concat([df_stats_boot, df_cis], axis=1)

    old_cols_order = df_stats_boot.columns.tolist()
    new_cols_order = list()
    prefix = [c.split("_")[0] for c in old_cols_order]
    prefix = list(dict.fromkeys(prefix))
    for c in prefix:
        labels = df_stats_boot.filter(regex=c, axis=1).columns.tolist()
        for i in labels:
            new_cols_order.append(i)

    df_stats_boot = df_stats_boot[new_cols_order]

    return df_stats_boot
