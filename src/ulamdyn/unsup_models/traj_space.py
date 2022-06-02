"""Base class and methods used to perform clustering analysis on the trajectory space."""
# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: May 27, 2022
import io
import numpy as np
from ulamdyn.unsup_models.utilities import Utils

try:
    import modin.pandas as pd
except ModuleNotFoundError:
    import pandas as pd

try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.utils import to_time_series_dataset
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    from tslearn.preprocessing import TimeSeriesScalerMinMax
except ModuleNotFoundError as e:
    print("Cannot import tslearn modules.")
    print("Please make sure that tslearn library has been installed.")
    print(e)

__all__ = ["ClusterTrajs"]


class ClusterTrajs(Utils):
    """Class used to find groups of similar trajectories in the molecular dynamics data."""

    def __str__(self) -> str:
        """Provide a string representation of the class.

        :return: Short description of the class functionality.
        :rtype: str
        """
        cls_status = "Clustering object used to group MD trajectories by similarity.\n"
        cls_status += "   Current status of the class variables:\n"
        cls_status += "  ----------------------------------------\n"
        for var in vars(self):
            if var != "data":
                cls_status += "     \u2022 {} ---> {}\n".format(var, getattr(self, var))
            else:
                if self.data is not None:
                    cls_status += "     \u2022 Size of loaded dataset ---> {}\n".format(
                        self.data.shape
                    )
                    buf = io.StringIO()
                    self.data.info(buf=buf)
                    data_info = buf.getvalue()
                    cls_status += data_info
                else:
                    cls_status += "     \u2022 The dataset variable is empty."
        return cls_status

    def __init__(
        self, data, dt=None, scaler=None, random_state=42, n_cpus=-1, verbosity=1
    ):
        """Class initializer for Clustering methods."""
        # Data must be a dataframe object including the TRAJ and time columns
        self.data = self._filter_by_dt(data, dt)
        self.id_trajs = self.data["TRAJ"].unique().tolist()
        self.scaler = scaler
        self.random_state = random_state
        self.n_cpus = n_cpus
        self.verbosity = verbosity

    def transform(self):
        all_trajs = []
        for id in self.id_trajs:
            trj = (
                self.data[self.data["TRAJ"] == id].drop(["TRAJ", "time"], axis=1).values
            )
            all_trajs.append(trj)
        all_trajs = to_time_series_dataset(all_trajs)

        sc_option = {
            "minmax": TimeSeriesScalerMinMax(value_range=(0.0, 1.0)),
            "standard": TimeSeriesScalerMeanVariance(mu=0.0, std=1.0),
        }

        if self.scaler in sc_option.keys():
            all_trajs = sc_option.get(self.scaler).fit_transform(all_trajs)

        return all_trajs

    def _run_model(self, model, X):

        self._print_model_params(model)

        model.fit(X)
        cluster_labels = model.labels_
        model_name = [type(model).__name__.lower() + "_labels"]
        df = pd.DataFrame({"TRAJ": self.id_trajs, model_name: cluster_labels})

        cluster_count = df.groupby(model_name).size().reset_index().values

        print(36 * "_")
        print(" Number of trajectories per cluster:\n")
        for n, size in cluster_count:
            print("       cluster {} ---> {:<6}".format(str(n), str(size)))
        print(36 * "_")
        print(" ")

        return df

    def kmeans(
        self,
        n_clusters=3,
        metric="dtw",
        metric_params=None,
        n_init=5,
        max_iter=100,
        convergence=1e-6,
        save_model=True,
    ):
        X_train = self.transform()
        print("\n***********************************************")
        print("*  Starting the K-Means clustering analysis:  *")
        print("***********************************************\n")

        model = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric=metric,
            metric_params=metric_params,
            tol=convergence,
            n_init=n_init,
            max_iter=max_iter,
            n_jobs=self.n_cpus,
            random_state=self.random_state,
            verbose=self.verbosity,
        )

        df_labels = self._run_model(model, X_train)
        if len(df_labels.index) > 0:
            self.data = self.data.merge(df_labels, on="TRAJ")

        if save_model:
            filename = "kmeans_model_trajs_nc" + str(n_clusters) + ".h5"
            model.to_hdf5(filename)

        return df_labels
