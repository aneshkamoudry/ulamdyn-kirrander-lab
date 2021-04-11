## Author: Max Pinheiro Jr <maxjr82@gmail.com>
## Date: 04/02/2021

import os
import sys
import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import StandardScaler

    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
except:
    print("Sklearn modules not found.")
    print("Please make sure that sklearn library has been installed.")

class DimensionalityReduction:

    def __str__(self):
         return "Unsupervised learning models for dimensionality reduction."

    def __init__ (self, data, n_samples=None, scaler=str(),
                  random_state=42, n_cpus=-1):
        self.df = data
        if n_samples is not None:
            self.df = data.sample(n_samples)
        self.indices = self.df.index.values
        self.n_cpus = n_cpus
        self.scaler = scaler.lower().strip()
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
    
    def pca(self, n_components=2, calc_error=False, save_errors=False):

        if self.scaler:
            data = self.data_scaling
        else:
            data = self.df.copy()
            warning_msg = "WARNING: The data will not be rescaled for PCA analysis!\n"
            warning_msg += "        The results are not reliable in this case.\n"
            print(warning_msg)

        model = PCA(n_components=n_components, svd_solver='full', 
                    random_state=self.random_state)

        print("************************************************")
        print("*  Starting the Principal Component Analysis:  *")
        print("************************************************\n")

        model.fit(data)
        X_transformed = model.transform(data)

        print("**********************************************")
        print("*  Percentage of variance explained by PCA:  *")
        print("**********************************************\n")
        explained_var = model.explained_variance_ratio_
        for count, value in enumerate(explained_var, start=1):
            print("     PC{}  -->  {:2.3f} %".format(count, value*100))
            if count > 5:
                break
        print("")

        col_labels = ['PC' + str(i+1) for i in range(X_transformed.shape[1])]
        df_transformed = pd.DataFrame(X_transformed, columns=col_labels) 
        df_transformed.index = self.indices

        if calc_error:
            X_reconstructed = model.inverse_transform(X_transformed)
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

        model.fit(data)
        X_transformed = model.transform(data)

        col_labels = ['KPC' + str(i+1) for i in range(X_transformed.shape[1])]
        df_transformed = pd.DataFrame(X_transformed, columns=col_labels) 
        df_transformed.index = self.indices

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

        model.fit(data)
        X_transformed = model.transform(data)

        col_labels = ['X' + str(i+1) for i in range(X_transformed.shape[1])]
        df_transformed = pd.DataFrame(X_transformed, columns=col_labels) 
        df_transformed.index = self.indices

        if calc_error:
            error = model.reconstruction_error()
            print("   Total reconstruction error = {:2.3f}".format(error))

        return df_transformed
