## Author: Max Pinheiro Jr <maxjr82@gmail.com>
## Date: April 2, 2021
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
    import ray

    ray.shutdown()
    ray.init()
except ModuleNotFoundError:
    import pandas as pd

try:
    from sklearn.base import clone
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import StandardScaler

    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.manifold import TSNE, Isomap, SpectralEmbedding

    from sklearn.cluster import KMeans
    from sklearn.cluster import SpectralClustering
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import calinski_harabasz_score
except ModuleNotFoundError as e:
    print("Required sklearn modules were not found.")
    print("Please make sure that sklearn library has been installed.")
    print(e)

__all__ = ["DimensionalityReduction", "Clustering"]


class Utils:
    @staticmethod
    def _data_scaling(scaler, df):

        sc_option = {
            "minmax": MinMaxScaler(),
            "standard": StandardScaler(),
            "robust": RobustScaler(),
        }

        if scaler not in sc_option.keys():
            return "Please choose a valid scaler: minmax, standard or robust."
        else:
            print(" ")
            print("Scaling data with {} method".format(scaler))
            scaled_data = sc_option.get(scaler).fit_transform(df)
            df_scaled = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
            print(" ")
            return df_scaled

    @staticmethod
    def _print_model_params(model):
        print(" The following set of parameters will be used:\n")
        for k, v in model.__dict__.items():
            print(" {:>20} = {:<20}".format(k, str(v)))
        print(" ")


class DimensionalityReduction(Utils):
    """Class used to find a low dimensional representation of MD trajectories data."""

    def __repr__(self) -> str:
        """Provide a string representation of the class.

        :return: Short description of the class functionality.
        :rtype: str
        """
        return "Unsupervised learning methods for dimensionality reduction."

    def __init__(self, data, n_samples=None, scaler=None, random_state=42, n_cpus=-1):
        """Class initializer for dimensionality reduction.

        :param data: Dataset of molecular geometries (or properties) extracted from the
                     available MD trajectories.
        :type data: pandas.DataFrame | modin.pandas.dataframe.DataFrame
        :param n_samples: If the value is not None, the dimensionality reduction analysis
                          will be performed on a randomly selected subsample of the original
                          dataset, defaults to None.
        :type n_samples: int, optional
        :param scaler: Define one of the three available methods (MinMax, Standard and Robust),
                       to rescale the original dataset before applying a dimensionality
                       reduction algorithm, defaults to None.
        :type scaler: str, optional
        :param random_state: Determines the random number generator for reproducible results
                             across multiple function calls, defaults to 42.
        :type random_state: int, optional
        :param n_cpus: Set up he number of parallel jobs to run the dimensionality reduction
                       methods. This parameter works only for the
                       :meth:`~ulamdyn.DimensionalityReduction.kpca`,
                       :meth:`~ulamdyn.DimensionalityReduction.isomap`, and
                       :meth:`~ulamdyn.DimensionalityReduction.tsne` methods,
                       defaults to -1 which means all processors will be used.
        :type n_cpus: int, optional
        """
        self.df = data
        self.indices = self.df.index.values

        if n_samples is not None:
            self._sample_data(n_samples)

        self.scaler = scaler
        if self.scaler is not None:
            self.df = self._data_scaling(self.scaler, self.df)

        self.random_state = random_state
        self.n_cpus = n_cpus

    def _sample_data(self, n_samples):
        self.df = self.df.sample(n_samples)
        self.indices = self.df.index.values

    def _run_model(self, model, data):

        self._print_model_params(model)

        model.fit(data)
        X_transformed = model.transform(data)

        col_labels = ["X" + str(i + 1) for i in range(X_transformed.shape[1])]
        df = pd.DataFrame(X_transformed, columns=col_labels)
        df.index = self.indices

        return df

    def pca(self, n_components=2, calc_error=False, save_errors=False):
        """Perform a linear dimensionality reduction using principal component analysis.

        .. note:: By default the percentage of variance explained by each of the selected
                  components will be printed after the PCA analysis.

        :param n_components: Number of principal components to keep, defaults to 2.
        :type n_components: int, optional.
        :param calc_error: If True, the reconstruction error between the original and the
                           projected data will be calculated, defaults to False.
        :type calc_error: bool, optional
        :param save_errors: If True, save to a csv file the reconstruction error calculated
                            for each sample, defaults to False.
        :type save_errors: bool, optional
        :return: a new dataset with the transformed values where the selected components
                 are stored in columns.
        :rtype: pandas.DataFrame | modin.pandas.dataframe.DataFrame
        """
        if not self.scaler:
            warning_msg = "---------------------------------------------------\n"
            warning_msg += "WARNING:\n"
            warning_msg += "The input data has not been standardized!\n"
            warning_msg += "This may lead to unreliable results for PCA.\n"
            warning_msg += "---------------------------------------------------\n"
            print(warning_msg)

        model = PCA(
            n_components=n_components, svd_solver="full", random_state=self.random_state
        )

        print("************************************************")
        print("*  Starting the Principal Component Analysis:  *")
        print("************************************************\n")

        df_transformed = self._run_model(model, self.df)

        print("**********************************************")
        print("*  Percentage of variance explained by PCA:  *")
        print("**********************************************\n")
        explained_var = model.explained_variance_ratio_
        for count, value in enumerate(explained_var, start=1):
            print("     PC{}  -->  {:2.3f} %".format(count, value * 100))
            if count > 5:
                break
        print("")

        if calc_error:
            X_reconstructed = model.inverse_transform(df_transformed.values)
            squared_errors = (self.df - X_reconstructed) ** 2
            col_labels = ["SE" + str(i + 1) for i in range(squared_errors.shape[1])]
            if save_errors:
                df_errors = pd.DataFrame(squared_errors, columns=col_labels)
                df_errors.index = self.indices
                df_errors.to_csv("pca_reconstruction_error.csv", index=True)

            print("***********************************")
            print("*    PCA reconstruction error:    *")
            print("***********************************\n")
            # Calculate the mean over the columns
            rmse_vals = np.sqrt(squared_errors.mean(axis=0))
            for count, rmse in enumerate(rmse_vals, start=1):
                print("     RMSE_PC{} = {:2.3f} ".format(count, rmse))
            tot_rmse = np.sqrt(np.mean(squared_errors))
            print(" ")
            print("     Total RMSE = {:2.3f} ".format(tot_rmse))
            print(" ")

        return df_transformed

    def kpca(
        self,
        n_components=2,
        kernel="rbf",
        gamma=None,
        degree=4,
        coef0=1,
        kernel_params=None,
        alpha=1.0,
        fit_inverse_transform=False,
    ):
        """Perform a nonlinear dimensionality reduction using kernel PCA.

        :param n_components: Number of components (features) to keep after KPCA
                             transformation, defaults to 2.
        :type n_components: int, optional
        :param kernel: Kernel function used in the transformation. The possible values are
                       'linear', 'poly', 'rbf', 'sigmoid', 'cosine' or precomputed',
                       defaults to "rbf".
        :type kernel: str, optional
        :param gamma: Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other
                      kernels. If gamma is None, then it is set to 1/n_features.
                      Defaults to None.
        :type gamma: float, optional
        :param degree: Degree of polynomial kernel. Ignored by other kernels. Defaults to 4.
        :type degree: int, optional
        :param coef0: Independent term in poly and sigmoid kernels. Ignored by other kernels.
                      Defaults to 1.
        :type coef0: int, optional
        :param kernel_params: Parameters (keyword arguments) and values for kernel passed as
                              callable object. Ignored by other kernels. Defaults to None.
        :type kernel_params: dict, optional
        :param alpha: Hyperparameter of the ridge regression that learns the inverse transform
                      (when fit_inverse_transform=True), defaults to 1.0.
        :type alpha: float, optional
        :param fit_inverse_transform: Hyperparameter of the ridge regression that learns the
                                      inverse transform (when fit_inverse_transform=True),
                                      defaults to False.
        :type fit_inverse_transform: bool, optional
        :return: a new dataset with the transformed values where the selected components
                 are stored in columns.
        :rtype: pandas.DataFrame | modin.pandas.dataframe.DataFrame
        """
        model = KernelPCA(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            coef0=coef0,
            degree=degree,
            alpha=alpha,
            kernel_params=kernel_params,
            fit_inverse_transform=fit_inverse_transform,
            random_state=self.random_state,
            n_jobs=self.n_cpus,
        )

        print("***************************************")
        print("*  Starting the Kernel PCA analysis:  *")
        print("***************************************\n")

        df_transformed = self._run_model(model, self.df)

        return df_transformed

    def isomap(
        self,
        n_components=2,
        n_neighbors=10,
        neighbors_algorithm="auto",
        metric="minkowski",
        p=2,
        metric_params=None,
        calc_error=False,
    ):
        """Perform a nonlinear dimensionality reduction through Isometric Mapping.

        :param n_components: Number of coordinates (features) for the low-dimensional
                             manifold, defaults to 2.
        :type n_components: int, optional
        :param n_neighbors: Number of neighbors to consider around each point,
                            defaults to 10.
        :type n_neighbors: int, optional
        :param neighbors_algorithm: Method used for nearest neighbors search,
                                    defaults to "auto"
        :type neighbors_algorithm: str, optional
        :param metric: The metric to use when calculating distance between instances in
                       a feature array. If metric is a string or callable, it must be one
                       of the options allowed by sklearn.metrics.pairwise_distances for its
                       metric parameter. If metric is “precomputed”, X is assumed to be a
                       distance matrix and must be square. Defaults to "minkowski".
        :type metric: str or callable, optional
        :param p: Parameter for the Minkowski metric from
                  sklearn.metrics.pairwise pairwise_distances. When p = 1, this is equivalent
                  to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
                  For arbitrary p, minkowski_distance (l_p) is used. Defaults to 2.
        :type p: int, optional
        :param metric_params: Additional keyword arguments for the metric function.
                              Defaults to None.
        :type metric_params: dict, optional
        :param calc_error: If True, the reconstruction error between the original and the
                           projected data will be calculated, defaults to False.
        :type calc_error: bool, optional
        :return: a new dataset with the transformed values where the coordinates of the
                 low-dimensional manifold are stored in columns.
        :rtype: pandas.DataFrame | modin.pandas.dataframe.DataFrame
        """
        model = Isomap(
            n_components=n_components,
            n_neighbors=n_neighbors,
            neighbors_algorithm=neighbors_algorithm,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=self.n_cpus,
        )

        print("***********************************")
        print("*  Starting the Isomap analysis:  *")
        print("***********************************\n")

        df_transformed = self._run_model(model, self.df)

        if calc_error:
            error = model.reconstruction_error()
            print(" Total reconstruction error = {:2.3f}\n".format(error))

        return df_transformed

    def tsne(
        self,
        n_components=2,
        perplexity=40.0,
        learning_rate=200.0,
        n_iter=2000,
        n_iter_without_progress=400,
        metric="euclidean",
        init="pca",
        verbose=1,
        method="barnes_hut",
    ):
        """Perform the t-distributed Stochastic Neighbor Embedding analysis.

        :param n_components: Number of coordinates (features) for the low-dimensional
                             embbeding, defaults to 2.
        :type n_components: int, optional
        :param perplexity: This hyperparameter is used to control the attention between local
                           and global aspects of the data, in a certain sense, by guessing the
                           number of close neighbors each point has. Larger datasets usually
                           require a larger perplexity. Consider selecting a value between 5
                           and 50. Different values can result in significantly different
                           results. Defaults to 40.0.
        :type perplexity: float, optional
        :param learning_rate: The learning rate for t-SNE is usually in the range [10.0,
                              1000.0]. If the learning rate is too high, the data may look
                              like a ‘ball’ with any point approximately equidistant from its
                              nearest neighbours. If the learning rate is too low, most points
                              may look compressed in a dense cloud with few outliers. If the
                              cost function gets stuck in a bad local minimum increasing the
                              learning rate may help. Defaults to 200.0
        :type learning_rate: float, optional
        :param n_iter: Maximum number of iterations for the optimization. Should be at least 250.
                       Defaults to 2000.
        :type n_iter: int, optional
        :param n_iter_without_progress: Maximum number of iterations without progress before we
                                        abort the optimization, used after 250 initial iterations
                                        with early exaggeration. Note that progress is only checked
                                        every 50 iterations so this value is rounded to the next
                                        multiple of 50. Defaults to 400.
        :type n_iter_without_progress: int, optional
        :param metric: The metric to use when calculating distance between instances in a feature
                       array. If metric is a string, it must be one of the options allowed by scipy.
                       spatial.distance.pdist for its metric parameter, or a metric listed in
                       pairwise.PAIRWISE_DISTANCE_FUNCTIONS. If metric is “precomputed”, X is
                       assumed to be a distance matrix. Alternatively, if metric is a callable
                       function, it is called on each pair of instances (rows) and the resulting
                       value recorded. The callable should take two arrays from X as input and
                       return a value indicating the distance between them. The default is
                       “euclidean” which is interpreted as squared euclidean distance.
                       Defaults to "euclidean".
        :type metric: str or callable, optional
        :param init: Initialization of embedding. Possible options are 'random', 'pca', and a
                     numpy array of shape (n_samples, n_components). PCA initialization cannot
                     be used with precomputed distances and is usually more globally stable than
                     random initialization. Defaults to "pca".
        :type init: str, optional
        :param verbose: Verbosity level. Defaults to 1
        :type verbose: int, optional
        :param method: By default the gradient calculation algorithm uses Barnes-Hut
                       approximation running in O(NlogN) time. method=’exact’ will run on the
                       slower, but exact, algorithm in O(N^2) time. The exact algorithm should be
                       used when nearest-neighbor errors need to be better than 3%. However, the
                       exact method cannot scale to millions of examples. Defaults to "barnes_hut".
        :type method: str, optional
        :return: a new dataset with the transformed values where the coordinates of the
                 low-dimensional manifold are stored in columns.
        :rtype: pandas.DataFrame | modin.pandas.dataframe.DataFrame
        """
        model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            n_iter_without_progress=n_iter_without_progress,
            metric=metric,
            init=init,
            verbose=verbose,
            method=method,
            n_jobs=self.n_cpus,
            random_state=self.random_state,
        )

        print("*******************************************")
        print("*  Starting the t-SNE manifold analysis:  *")
        print("*******************************************\n")

        df_transformed = self._run_model(model, self.df)

        return df_transformed


class Clustering(Utils):
    def __str__(self):
        return "Clustering methods."

    def __init__(
        self, data, n_samples=None, scaler=None, random_state=51, n_cpus=-1, verbosity=0
    ):
        self.df = data
        if n_samples is not None:
            self.df = data.sample(n_samples)
        self.indices = self.df.index.values

        self.scaler = scaler
        if self.scaler is not None:
            self.df = self._data_scaling(self.scaler, self.df)

        self.random_state = random_state
        self.n_cpus = n_cpus
        self.verbosity = verbosity

    def _run_model(self, model, data):

        self._print_model_params(model)

        model.fit(data)

        col_name = [type(model).__name__.lower() + "_labels"]
        df = pd.DataFrame(model.labels_, columns=col_name)
        df.index = self.indices

        cluster_count = df.groupby(col_name).size().reset_index().values

        print(36 * "_")
        print(" Number of geometries per cluster:\n")
        for n, size in cluster_count:
            print("       cluster {} ---> {:<6}".format(str(n), str(size)))
        print(36 * "_")
        print(" ")

        return df

    def _opt_num_clusters(self, model):

        df_temp = self.df.copy(deep=True)
        if df_temp.shape[0] > 5000:
            df_temp = df_temp.sample(5000)

        if isinstance(model.n_clusters, list):
            range_n_clusters = np.array(model.n_clusters)
        elif model.n_clusters == "best":
            range_n_clusters = np.arange(2, 13)

        print("Evaluate clustering performance:\n")
        scores_silhouette = list()
        scores_ch = list()
        for k in range_n_clusters:
            model.n_clusters = k
            fitted_model = model.fit(df_temp)
            cluster_labels = fitted_model.labels_

            silhouette_avg = silhouette_score(df_temp, cluster_labels)
            ch = calinski_harabasz_score(df_temp, cluster_labels)
            print("For n_clusters = {}".format(k))
            print("   the average silhouette score is {:2.4f}".format(silhouette_avg))
            print("   the Calinski and Harabasz score is {:2.4f}\n".format(ch))
            scores_silhouette.append(silhouette_avg)
            scores_ch.append(ch)

        scores_silhouette = np.array(scores_silhouette)
        idx_high_scores = np.argsort(scores_silhouette)[-4:]
        range_n_clusters = range_n_clusters[idx_high_scores]

        scores_ch = np.array(scores_ch)
        scores_ch = scores_ch[idx_high_scores]
        idx_best = np.argmax(scores_ch)
        best_n_clusters = range_n_clusters[idx_best]

        return best_n_clusters

    def kmeans(
        self,
        n_clusters=5,
        init="k-means++",
        n_init=500,
        max_iter=2000,
        convergence=1e-06,
    ):

        print("\n***********************************************")
        print("*  Starting the K-Means clustering analysis:  *")
        print("***********************************************\n")

        model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=convergence,
            random_state=self.random_state,
            verbose=self.verbosity,
        )

        if isinstance(model.n_clusters, list) or model.n_clusters == "best":
            clean_model = clone(model)
            print("Searching for the optimal number of clusters...\n")
            k_best = self._opt_num_clusters(model)
            clean_model.n_clusters = k_best
            model = clean_model
        else:
            print("ERROR:\n Invalid option!")
            return None

        df_labels = self._run_model(model, self.df)

        return df_labels

    def hierarchical(
        self,
        n_clusters=5,
        affinity="cosine",
        connectivity=None,
        compute_full_tree="auto",
        linkage="single",
        distance_threshold=None,
    ):

        if isinstance(distance_threshold, float):
            n_clusters = None

        print("\n*****************************************************")
        print("*  Starting the Agglomerative clustering analysis:  *")
        print("***************************************************\n")

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            distance_threshold=distance_threshold,
        )

        df_labels = self._run_model(model, self.df)

        if n_clusters is None:
            print("  Number of clusters found by the algorithm:\n")
            print("  n_clusters = {}".format(model.n_clusters_))

        print("  Number of leaves in the hierarchical tree:\n")
        print("  n_leaves = {}".format(model.n_leaves_))

        return df_labels

    def spectral(
        self,
        n_clusters=5,
        n_components=10,
        n_init=100,
        gamma=0.005,
        affinity="rbf",
        n_neighbors=20,
        degree=3,
        coef0=1,
        kernel_params=None,
    ):

        print("\n************************************************")
        print("*  Starting the Spectral clustering analysis:  *")
        print("************************************************\n")

        model = SpectralClustering(
            n_clusters=n_clusters,
            n_components=n_components,
            n_init=n_init,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            n_neighbors=n_neighbors,
            affinity=affinity,
            kernel_params=kernel_params,
            n_jobs=self.n_cpus,
        )

        if isinstance(model.n_clusters, list) or model.n_clusters == "best":
            clean_model = clone(model)
            print("Searching for the optimal number of clusters...\n")
            k_best = self._opt_num_clusters(model)
            clean_model.n_clusters = k_best
            model = clean_model
        else:
            print("ERROR:\n Invalid option!")
            return None

        df_labels = self._run_model(model, self.df)

        return df_labels
