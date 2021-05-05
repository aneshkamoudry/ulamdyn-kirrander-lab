## Author: Max Pinheiro Jr <maxjr82@gmail.com>
## Date: 04/02/2021

from __future__ import (absolute_import, division, print_function,
                        unicode_literals, with_statement)

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
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import StandardScaler

    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
except:
    print("Required sklearn modules were not found.")
    print("Please make sure that sklearn library has been installed.")

class DimensionalityReduction:

    def __str__(self):
         return "Unsupervised learning models for dimensionality reduction."

    def __init__ (self, data, n_samples=None, scaler=None,
                  random_state=42, n_cpus=-1):
        self.df = data
        if n_samples is not None:
            self.df = data.sample(n_samples)
        self.indices = self.df.index.values
        self.n_cpus = n_cpus
        self.scaler = scaler
        self.random_state = random_state
        
    @property
    def data_scaling(self):
        
        sc_option = {'minmax': MinMaxScaler(), 
                     'standard': StandardScaler(),
                     'robust': RobustScaler()}
    
        if self.scaler not in sc_option.keys():
            return "Please choose a valid scaler: minmax, standard or robust."
        else:
            df_scaled = sc_option.get(self.scaler).fit_transform(self.df)
            return df_scaled
    
    def _print_model_params(self,model):
        print(" The following parameters set will be used:\n")
        for k,v in model.__dict__.items():
            print(" {:>20} = {:<20}".format(k,str(v)))
        print(" ")

    def _run_model(self,model,data):

        model.fit(data)
        X_transformed = model.transform(data)

        col_labels = ['X' + str(i+1) for i in range(X_transformed.shape[1])]
        df = pd.DataFrame(X_transformed, columns=col_labels) 
        df.index = self.indices

        return df    

    def pca(self, n_components=2, calc_error=False, save_errors=False):

        if self.scaler:
            data = self.data_scaling
        else:
            data = self.df.values
            warning_msg = "---------------------------------------------------\n"
            warning_msg += "WARNING:\n"
            warning_msg += "The data will not be rescaled for PCA analysis!\n"
            warning_msg += "Results can be unreliable in this case.\n"
            warning_msg += "---------------------------------------------------\n"
            print(warning_msg)

        model = PCA(n_components=n_components, svd_solver='full', 
                    random_state=self.random_state)

        print("************************************************")
        print("*  Starting the Principal Component Analysis:  *")
        print("************************************************\n")

        self._print_model_params(model)

        df_transformed = self._run_model(model,data)
        
        print("**********************************************")
        print("*  Percentage of variance explained by PCA:  *")
        print("**********************************************\n")
        explained_var = model.explained_variance_ratio_
        for count, value in enumerate(explained_var, start=1):
            print("     PC{}  -->  {:2.3f} %".format(count, value*100))
            if count > 5:
                break
        print("")

        if calc_error:
            X_reconstructed = model.inverse_transform(df_transformed.values)
            squared_errors = (data - X_reconstructed)**2
            col_labels = ['SE' + str(i+1) for i in range(squared_errors.shape[1])]
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

    def kpca(self, n_components=2, kernel='rbf', gamma=None, degree=4, coef0=1, 
             kernel_params=None, alpha=1.0, fit_inverse_transform=False):

        if self.scaler:
            data = self.data_scaling
        else:
            data = self.df.copy()

        model = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, coef0=coef0,
                          degree=degree, alpha=alpha, kernel_params=kernel_params,
                          fit_inverse_transform=fit_inverse_transform, n_jobs=self.n_cpus,
                          random_state=self.random_state)

        print("***************************************")
        print("*  Starting the Kernel PCA analysis:  *")
        print("***************************************\n")

        self._print_model_params(model)

        df_transformed = self._run_model(model,data)

        return df_transformed

    def isomap(self, n_components=2, n_neighbors=10, neighbors_algorithm='auto', 
               metric='minkowski', p=2, metric_params=None, calc_error=False):

        if self.scaler:
            data = self.data_scaling
        else:
            data = self.df.copy()

        model = Isomap(n_components=n_components, n_neighbors=n_neighbors,
                       neighbors_algorithm=neighbors_algorithm, metric=metric,
                       p=p, metric_params=metric_params, n_jobs=self.n_cpus)

        print("***********************************")
        print("*  Starting the Isomap analysis:  *")
        print("***********************************\n")

        self._print_model_params(model)

        df_transformed = self._run_model(model,data)

        if calc_error:
            error = model.reconstruction_error()
            print(" Total reconstruction error = {:2.3f}\n".format(error))

        return df_transformed
    
    def tsne(self, n_components=2, perplexity=40.0, learning_rate=200.0, n_iter=2000, 
             n_iter_without_progress=400, metric='euclidean', init='pca', verbose=1, 
             method='barnes_hut', angle=0.5, square_distances='legacy'):
     
        if self.scaler:
            data = self.data_scaling
        else:
            data = self.df.copy()

        model = TSNE(n_components=n_components, perplexity=perplexity, 
                     learning_rate=learning_rate, n_iter=n_iter, 
                     n_iter_without_progress=n_iter_without_progress, 
                     metric=metric, init=init, verbose=verbose, method=method, 
                     angle=angle, square_distances=square_distances, 
                     n_jobs=self.n_cpus, random_state=self.random_state)

        print("*******************************************")
        print("*  Starting the t-SNE manifold analysis:  *")
        print("*******************************************\n")

        self._print_model_params(model)

        df_transformed = self._run_model(model,data)

        return df_transformed