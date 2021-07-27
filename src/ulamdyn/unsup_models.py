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

    ray.init()
except:
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
except:
    print("Required sklearn modules were not found.")
    print("Please make sure that sklearn library has been installed.")

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
    def __str__(self):
        return "Unsupervised learning models for dimensionality reduction."

    def __init__(self, data, n_samples=None, scaler=None, random_state=42, n_cpus=-1):
        self.df = data
        if n_samples is not None:
            self.df = data.sample(n_samples)
        self.indices = self.df.index.values

        self.scaler = scaler
        if self.scaler is not None:
            self.df = self._data_scaling(self.scaler, self.df)

        self.random_state = random_state
        self.n_cpus = n_cpus

    def _run_model(self, model, data):

        self._print_model_params(model)

        model.fit(data)
        X_transformed = model.transform(data)

        col_labels = ["X" + str(i + 1) for i in range(X_transformed.shape[1])]
        df = pd.DataFrame(X_transformed, columns=col_labels)
        df.index = self.indices

        return df

    def pca(self, n_components=2, calc_error=False, save_errors=False):

        if not self.scaler:
            warning_msg = "---------------------------------------------------\n"
            warning_msg += "WARNING:\n"
            warning_msg += "The data is not properly scaled!\n"
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
        """Isomap Embedding

        Non-linear dimensionality reduction through Isometric Mapping

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
        angle=0.5,
        square_distances="legacy",
    ):

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
            angle=angle,
            square_distances=square_distances,
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
        compute_distances=False,
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
            compute_distances=compute_distances,
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
