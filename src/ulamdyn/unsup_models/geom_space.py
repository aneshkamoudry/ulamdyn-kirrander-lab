"""Classes and methods used to perform dimension reduction and clustering on geometry space."""
# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: April 2, 2021
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
    with_statement,
)

import io
import numpy as np
from joblib import dump, load
from ulamdyn.unsup_models.utilities import Utils

try:
    import modin.pandas as pd

except ModuleNotFoundError:
    import pandas as pd

try:
    from sklearn.base import clone

    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.manifold import TSNE, Isomap

    from sklearn.cluster import KMeans
    from sklearn.cluster import SpectralClustering
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import calinski_harabasz_score
    from sklearn.mixture import GaussianMixture
except ModuleNotFoundError as e:
    print("Required sklearn modules were not found.")
    print("Please make sure that sklearn library has been installed.")
    print(e)

__all__ = ["DimensionReduction", "ClusterGeoms"]


class DimensionReduction(Utils):
    """Class used to find a low dimensional representation of MD trajectories data."""

    def __str__(self) -> str:
        """Provide a string representation of the class.

        :return: Short description of the class functionality.
        :rtype: str
        """
        return "Unsupervised learning methods for dimensionality reduction."

    def __init__(
        self, data, dt=None, n_samples=None, scaler=None, random_state=42, n_cpus=-1
    ) -> None:
        """Class initializer for dimensionality reduction.

        :param data: Dataset of molecular geometries (or properties) extracted from the
                     available MD trajectories.
        :type data: pandas.DataFrame | modin.pandas.dataframe.DataFrame
        :param dt: Time step used to filter the input data by multiple of dt before feeding
                          the data into the clustering algorithm, defaults to None.
        :type dt: float, optional
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
                       :meth:`~ulamdyn.DimensionReduction.kpca`,
                       :meth:`~ulamdyn.DimensionReduction.isomap`, and
                       :meth:`~ulamdyn.DimensionReduction.tsne` methods,
                       defaults to -1 which means all processors will be used.
        :type n_cpus: int, optional
        """
        self.df = self._filter_by_dt(data, dt)
        self.df = self.df.drop(["TRAJ", "time"], axis=1, errors="ignore")
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
        model_name = type(model).__name__.lower()

        if model_name == "tsne":
            X_transformed = model.fit_transform(data)
        else:
            model.fit(data)
            X_transformed = model.transform(data)

        col_labels = ["X" + str(i + 1) for i in range(X_transformed.shape[1])]
        df = pd.DataFrame(X_transformed, columns=col_labels)
        df.index = self.indices

        return df

    def _create_pca_importance(self, pca):
        df_importance = pd.DataFrame(pca.components_)
        df_importance.columns = self.df.columns
        df_importance = df_importance.apply(np.abs)
        df_importance = df_importance.transpose()
        num_pcs = df_importance.shape[1]

        # Generate column names
        col_names = [f"PC{i}" for i in range(1, num_pcs + 1)]
        df_importance.columns = col_names

        print("***********************************")
        print("*     PCA feature importance:     *")
        print("***********************************\n")

        for pc in df_importance.columns:
            top_5_features = df_importance[pc].sort_values(ascending=False)[:5]
            print(), print("Top 5 features for {} \n".format(pc))
            print(top_5_features)
        print()

        return df_importance

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

        df_importance = self._create_pca_importance(model)
        df_importance.to_csv("pca_feature_importance.csv")

        if calc_error:
            X_reconstructed = model.inverse_transform(df_transformed.values)
            squared_errors = (self.df.values - X_reconstructed) ** 2
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
        n_neighbors=30,
        neighbors_algorithm="auto",
        metric=None,
        p=2,
        metric_params=None,
        calc_error=False,
    ):
        """Perform a nonlinear dimensionality reduction through Isometric Mapping.

        :param n_components: Number of coordinates (features) for the low-dimensional
                             manifold, defaults to 2.
        :type n_components: int, optional
        :param n_neighbors: Number of neighbors to consider around each point,
                            defaults to 12.
        :type n_neighbors: int, optional
        :param neighbors_algorithm: Method used for nearest neighbors search,
                                    defaults to "auto"
        :type neighbors_algorithm: str, optional
        :param metric: The metric to use when calculating distance between instances in
                       a feature array. If metric is a string or callable, it must be one
                       of the options allowed by sklearn.metrics.pairwise_distances for its
                       metric parameter. If metric is “precomputed”, X is assumed to be a
                       distance matrix and must be square. Defaults to "cosine".
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
        if metric is None:
            metric = "cosine"

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
        learning_rate=180.0,
        n_iter=2000,
        n_iter_without_progress=400,
        metric="euclidean",
        init="pca",
        verbose=1,
        method="barnes_hut",
    ):
        """Perform the t-distributed Stochastic Neighbor Embedding analysis.

        :param n_components: Number of coordinates (features) for the low-dimensional
                             embedding, defaults to 2.
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


class ClusterGeoms(Utils):
    """Class used to find groups of similar geometries in the MD trajectories data."""

    def __str__(self) -> str:
        """Provide a string representation of the class.

        :return: Short description of the class functionality.
        :rtype: str
        """
        cls_status = "Class used to group molecular geometries by similarity.\n"
        methods_list = [
            func
            for func in dir(self)
            if callable(getattr(self, func)) and not func.startswith("_")
        ]
        cls_status += "List of available methods: {}\n".format(" ".join(methods_list))
        cls_status += "   Current state of the class variables:\n"
        cls_status += "  ---------------------------------------\n"
        cls_status += "   \u2022 data scaler method = {}\n".format(self.scaler)
        cls_status += "   \u2022 random state = {}\n".format(self.random_state)
        cls_status += "   \u2022 number of cpus = {}\n".format(self.n_cpus)
        cls_status += "   \u2022 verbosity level = {}\n".format(self.verbosity)
        if self.df is not None:
            cls_status += "   \u2022 Size of the input data = {}\n".format(
                self.df.shape
            )
            buf = io.StringIO()
            self.df.info(buf=buf)
            data_info = buf.getvalue()
            cls_status += data_info
        return cls_status

    def __init__(
        self,
        data,
        dt=None,
        indices=None,
        n_samples=None,
        scaler=None,
        random_state=51,
        n_cpus=-1,
        verbosity=0,
    ):
        """Class initializer for Clustering methods.

        :param data: Dataset with all molecular geometries (or QM properties) extracted from
                     the loaded MD trajectories.
        :type data: pandas.DataFrame | modin.pandas.dataframe.DataFrame
        :param dt: Time step used to filter the input data by multiple of dt before feeding
                          the data into the clustering algorithm, defaults to None.
        :type dt: float, optional
        :param indices: Name of a text file containing the list of indices as a single column
                          used to filter the input data, defaults to None.
        :type indices: str, optional
        :param n_samples: Size of the subsample selected randomly from the original data to
                          perform the clustering analysis, defaults to None.
        :type n_samples: int, optional
        :param scaler: Define one of the three available methods (MinMax, Standard and Robust),
                       to rescale the original dataset before applying the Clustering
                       algorithm, defaults to None.
        :type scaler: str, optional
        :param random_state: Determines the random number generator for reproducible results
                             across multiple function calls, defaults to 42.
        :type random_state: int, optional
        :param n_cpus: Set up he number of parallel jobs to run the clustering analysis,
                       defaults to -1.
        :type n_cpus: int, optional
        :param verbosity: Control the level of printed information, defaults to 0.
        :type verbosity: int, optional
        """
        if indices is None:
            self.df = self._filter_by_dt(data, dt)
        else:
            self.indices = np.loadtxt(indices)
            self.df = self.df.loc[self.indices]

        self.df = self.df.drop(["TRAJ", "time"], axis=1, errors="ignore")

        if n_samples is not None:
            if n_samples > self.df.shape[0]:
                n_samples = self.df.shape[0]
            self.df = self.df.sample(n_samples)

        self.indices = self.df.index.values

        self.scaler = scaler
        if self.scaler is not None:
            self.df = self._data_scaling(self.scaler, self.df)

        self.random_state = random_state
        self.n_cpus = n_cpus
        self.verbosity = verbosity
        self.model = None

    def _run_model(self, data):
        if isinstance(self.model.n_clusters, list) or self.model.n_clusters == "best":
            clean_model = clone(self.model)
            print("Searching for the optimal number of clusters...\n")
            k_best = self._opt_num_clusters(self.model)
            clean_model.n_clusters = k_best
            self.model = clean_model
        elif isinstance(self.model.n_clusters, int):
            pass
        else:
            print("ERROR:\n Invalid option!")
            return

        self._print_model_params(self.model)

        self.model.fit(data)

        model_name = type(self.model).__name__.lower()
        cluster_labels = self.model.labels_
        col_name = [model_name + "_labels"]
        df = pd.DataFrame(cluster_labels, columns=col_name)
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
        if df_temp.shape[0] > 8000:
            df_temp = df_temp.sample(8000)

        if isinstance(model.n_clusters, list):
            range_n_clusters = np.array(model.n_clusters)
        elif model.n_clusters == "best":
            range_n_clusters = np.arange(2, 12)

        print("Evaluate clustering performance:\n")
        scores_silhouette = []
        scores_ch = []
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

        print("The optimal number of clusters is {}\n".format(best_n_clusters))

        return best_n_clusters

    def kmeans(
        self,
        n_clusters=5,
        init="k-means++",
        n_init=100,
        max_iter=1000,
        convergence=1e-06,
        save_model=True,
    ):
        """Perform K-Means clustering in geometry space.

        :param n_clusters: The number of clusters to form that corresponds also to the
                           number of cluster centroids to generate, defaults to 5.
                           If a list is passed, the k-means algorithm will be run for all
                           n_clusters in the list, whereas if the argument is equal to
                           'best', consecutive runs will be performed with n_clusters
                           varying in the range of [2, 15]. In both cases, the final results
                           will be the best output labels with respect to the clustering
                           performance on the silhouette and Calinski-Harabasz scores.
        :type n_clusters: int, list or str optional
        :param init: Method for initialization :
                     + 'k-means++' -> selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
                     + 'random' -> choose n_clusters observations (rows) at random from data for the initial centroids.
                     + If an array is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
                     Defaults to "k-means++".
        :type init: str or array, optional
        :param n_init: Number of time the k-means algorithm will be run with different
                       centroid seeds. The final results will be the best output of n_init
                       consecutive runs in terms of loss function, defaults to 500.
        :type n_init: int, optional
        :param max_iter: Maximum number of iterations of the k-means algorithm for a single
                         run, defaults to 1000.
        :type max_iter: int, optional
        :param convergence: Relative tolerance with regards to Frobenius norm of the
                            difference in the cluster centers of two consecutive iterations
                            to declare convergence, defaults to 1e-06.
        :type convergence: float, optional
        :param save_model: Store the trained parameters of the model in a binary file,
                           defaults to True.
        :type save_model: bool, optional
        :return: Dataframe of shape (n_samples,) with cluster labels for each data point.
        :rtype: pandas.DataFrame | modin.pandas.dataframe.DataFrame
        """
        print("***********************************************")
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

        self.model = model

        df_labels = self._run_model(self.df)

        if save_model:
            filename = "kmeans_model_geoms_nc" + str(self.model.n_clusters) + ".joblib"
            dump(self.model, filename)

        return df_labels

    def gaussian_mixture(
        self,
        n_clusters=5,
        covariance="full",
        tol=0.0001,
        n_init=10,
        max_iter=500,
        init="k-means++",
        save_model=True,
    ):
        """Perform probabilist clustering in geometry space with Gaussian Mixture model.

        :param n_clusters: The number of clusters to find, which corresponds to the
                           number of mixed gaussians, defaults to 5.
        :type n_clusters: int, optional
        :param covariance: String describing the type of covariance parameters to use,
                           default is "full". Acceptable values are "full", "tied",
                           "diag", "spherical" (equivalent to K-Means).
        :type covariance: str, optional
        :param tol: Convergence criteria of the lower bound average gain, below which
                    the EM iterations stop, defaults to 1e-4.
        :type tol: float, optional
        :param n_init: The number of initializations to perform, where best results
                       are kept. The default is 10.
        :type n_init: int, optional
        :param max_iter: The number of EM iterations to perform, defaults to 500.
        :type max_iter: int, optional
        :param init: The method used to initialize the weights, the means and the
                     precisions, default is "k-means++". Acceptable strings are "k-means",
                     "k-means++", "random", or "‘random_from_data".
        """
        print("***********************************************")
        print("*    Starting the GMM clustering analysis:    *")
        print("***********************************************\n")

        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance,
            tol=tol,
            n_init=n_init,
            max_iter=max_iter,
            init_params=init,
            random_state=self.random_state,
            verbose=self.verbosity,
        )

        self.model = model
        self._print_model_params(self.model)

        labels = self.model.fit_predict(self.df)
        df_labels = pd.DataFrame(labels, columns=["gmm_labels"])
        probabilities = self.model.predict_proba(self.df)
        col_name = [f"Prob_C{i}" for i in range(n_clusters)]
        df_prob = pd.DataFrame(probabilities, columns=col_name)
        df_labels = pd.concat([df_labels, df_prob], axis=1)
        df_labels.index = self.indices

        cluster_count = df_labels.groupby(["gmm_labels"]).size().reset_index().values

        print(36 * "_")
        print(" Number of geometries per cluster:\n")
        for n, size in cluster_count:
            print("       cluster {} ---> {:<6}".format(str(n), str(size)))
        print(36 * "_")
        print(" ")

        if save_model:
            filename = "gm_model_geoms_nc" + str(self.model.n_components) + ".joblib"
            dump(self.model, filename)

        return df_labels

    def hierarchical(
        self,
        n_clusters=5,
        affinity="euclidean",
        connectivity=None,
        linkage="complete",
        distance_threshold=None,
        save_model=True,
    ):
        """Perform a hierarchical cluster analysis based on the agglomerative.

        :param n_clusters: The number of clusters to find. It must be None if
                           distance_threshold is not None, defaults to 5.
        :type n_clusters: int, optional
        :param affinity: Metric used to compute the linkage. Can be "euclidean", "l1",
                         "l2", "manhattan", "cosine", or "precomputed". If linkage is "ward",
                         only "euclidean" is accepted. If "precomputed", a distance matrix
                         (instead of a similarity matrix) is needed as input for the fit
                         method, defaults to "euclidean".
        :type affinity: str, optional
        :param connectivity: Connectivity matrix. Defines for each sample the neighboring
                             samples following a given structure of the data. This can be a
                             connectivity matrix itself or a callable that transforms the data
                             into a connectivity matrix, such as derived from kneighbors_graph.
                             Default is None, i.e, the hierarchical clustering algorithm is
                             unstructured.
        :type connectivity: array-like or callable, optional
        :param linkage: Define the linkage criterion to build the tree. It determines which
                        distance to use between sets of observation. The algorithm will merge
                        the pairs of cluster that minimize this criterion. The options are :
                        + 'ward' -> minimizes the variance of the clusters being merged.
                        + 'average' -> uses the average of the distances of each observation of the two sets.
                        + 'complete' or 'maximum' -> uses the maximum distances between all observations of the two sets.
                        + 'single' -> uses the minimum of the distances between all observations of the two sets.
                        The default is "complete".
        :type linkage: str, optional
        :param distance_threshold: The linkage distance threshold above which, clusters will not
                                   be merged. If not None, n_clusters must be None and
                                   compute_full_tree must be True, defaults to None.
        :type distance_threshold: float, optional
        :param save_model: Store the trained parameters of the model in a binary file,
                           defaults to True.
        :type save_model: bool, optional
        :return: Dataframe of shape (n_samples,) with cluster labels for each data point.
        :rtype: pandas.DataFrame | modin.pandas.dataframe.DataFrame
        """
        if isinstance(distance_threshold, float):
            n_clusters = None

        print("*****************************************************")
        print("*  Starting the Agglomerative clustering analysis:  *")
        print("*****************************************************\n")

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            connectivity=connectivity,
            linkage=linkage,
            distance_threshold=distance_threshold,
        )

        self.model = model

        df_labels = self._run_model(self.df)

        if n_clusters is None:
            print("  Number of clusters found by the algorithm:\n")
            print("  n_clusters = {}".format(model.n_clusters_))

        print("  Number of leaves in the hierarchical tree:\n")
        print("  n_leaves = {}".format(model.n_leaves_))

        if save_model:
            filename = "hierarchical_nc" + str(self.model.n_clusters) + ".joblib"
            dump(model, filename)

        return df_labels

    def spectral(
        self,
        n_clusters=5,
        n_components=10,
        n_init=100,
        affinity="rbf",
        gamma=0.01,
        n_neighbors=10,
        degree=3,
        coef0=1,
        kernel_params=None,
        save_model=True,
    ):
        """Apply clustering to a projection of the normalized Laplacian.

        Note that Spectral Clustering is a highly expensive method due to the computation
        of the affinity matrix. Hence, this method is recommended only for small to medium
        size datasets (n_samples < 10000).

        .. note:: This method is equivalent to kernel k-means (DOI: 10.1145/1014052.1014118).
                  Spectral clustering is recommended for non-linearly separable dataset,
                  where the individual clusters have a highly non-convex shape.

        :param n_clusters: The number of clusters to form which in this case corresponds
                           to the dimension of the projection subspace. The default is 5.

                           If a list is passed, the k-means algorithm will be run for all
                           n_clusters in the list, whereas if the argument is equal to
                           'best', consecutive runs will be performed with n_clusters
                           varying in the range of [2, 15]. In both cases, the final results
                           will be the best output labels with respect to the clustering
                           performance on the silhouette and Calinski-Harabasz scores.
        :type n_clusters: int, optional
        :param n_components: Number of eigenvectors to use for the spectral embedding,
                             defaults to 10
        :type n_components: int, optional
        :param n_init: Number of time the k-means algorithm will be run with different centroid
                       seeds. The final results will be the best output of n_init consecutive
                       runs in terms of inertia. Only used if assign_labels='kmeans'. The
                       default is 100.
        :type n_init: int, optional
        :param affinity: Method used to construct the affinity matrix. The available options are :
                         + 'nearest_neighbors': construct the affinity matrix by computing a graph of nearest neighbors.
                         + 'rbf': construct the affinity matrix using a radial basis function (RBF) kernel.
                         + 'precomputed_nearest_neighbors': interpret X as a sparse graph of precomputed distances, and construct a binary affinity matrix from the n_neighbors nearest neighbors of each instance.
                         + one of the kernels supported by pairwise_kernels.
                         The default method is "rbf".
        :type affinity: str or callable, optional
        :param gamma: Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels.
                      Ignored for affinity='nearest_neighbors'. Defaults to 0.01.
        :type gamma: float, optional
        :param n_neighbors: Number of neighbors to use when constructing the affinity matrix
                            using the nearest neighbors method. Ignored for affinity='rbf',
                            defaults to 20.
        :type n_neighbors: int, optional
        :param degree: Degree of the polynomial kernel. Ignored by other kernels. Defaults to 3.
        :type degree: int, optional
        :param coef0: Zero coefficient for polynomial and sigmoid kernels. Ignored by other
                      kernels. Defaults to 1.
        :type coef0: int, optional
        :param kernel_params: Parameters (keyword arguments) and values for kernel passed as
                              callable object. Ignored by other kernels. Defaults to None.
        :param save_model: Store the trained parameters of the model in a binary file,
                           defaults to True.
        :type save_model: bool, optional
        :type kernel_params: dict or str, optional
        :return: Dataframe of shape (n_samples,) with cluster labels for each data point.
        :rtype: pandas.DataFrame | modin.pandas.dataframe.DataFrame
        """
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

        self.model = model

        df_labels = self._run_model(self.df)

        if save_model:
            filename = "spectral_nc" + str(self.model.n_clusters) + ".joblib"
            dump(model, filename)

        return df_labels
