"""Auxiliary module to handle data preprocessing and printing."""

# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: May 31, 2022

try:
    import modin.pandas as pd

except ModuleNotFoundError:
    import pandas as pd

try:
    from sklearn.preprocessing import (
        MinMaxScaler,
        Normalizer,
        RobustScaler,
        StandardScaler,
    )
except ModuleNotFoundError as e:
    print("Required sklearn modules were not found.")
    print("Please make sure that sklearn library has been installed.")
    print(e)

__all__ = ["Utils"]


class Utils:
    @staticmethod
    def _data_scaling(scaler, df):

        sc_option = {
            "minmax": MinMaxScaler(),
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "norm": Normalizer(),
        }

        if scaler not in sc_option:
            err_msg = "ERROR! Please choose a valid scaler: "
            err_msg += "minmax, standard, robust or norm."
            print(err_msg)
            return

        print("\nScaling data with {} method\n".format(scaler))
        scaled_data = sc_option.get(scaler).fit_transform(df)
        df_scaled = pd.DataFrame(
            scaled_data, index=df.index, columns=df.columns
        )
        return df_scaled

    @staticmethod
    def _print_model_params(model):
        print(" The following set of parameters will be used:\n")
        for k, v in model.__dict__.items():
            print("   {:>20} = {:<20}".format(k, str(v)))
        print(" ")

    @staticmethod
    def _filter_by_dt(data, dt):
        if dt is not None:
            filter = ((data["time"] % dt).round(1) == 0.0) | (
                (data["time"] % dt).round(1) == dt
            )
            data = data[filter]
        return data
