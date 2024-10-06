# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: June 7, 2022

from ulamdyn.statistics import bootstrap, create_bootstrap_stats
from ulamdyn.wrappers._auxiliary import get_properties_data

__all__ = ["Bootstrap"]


class Bootstrap:
    @classmethod
    def _load_params(cls, **kw):
        cls.boot_options = kw.get("bootstrap")
        if cls.boot_options is not None:
            cls.boot_options = cls.boot_options.lower().strip()
            cls.boot_options = cls.boot_options.split(",")

    @classmethod
    def _load_properties(cls):
        print("Loading the all properties data...\n")
        cls.df_props = get_properties_data()

    @classmethod
    def _set_boot_params(cls):
        # Setting default values for the bootstrap statistics
        cls.save_csv = False
        cls.n_samples = len(cls.df_props["TRAJ"].unique())
        cls.n_repeats = 1000
        cls.ci_level = 95

        if cls.boot_options[0] != "save":
            cls.n_repeats = int(cls.boot_options[0])
            cls.boot_options = cls.boot_options[1:]
            if "save" in cls.boot_options:
                cls.save_csv = True
                cls.boot_options = [i for i in cls.boot_options if i != "save"]
            if len(cls.boot_options) > 0:
                try:
                    cls.n_samples, cls.ci_level = list(
                        map(int, cls.boot_options)
                    )
                except ValueError:
                    cls.ci_level = int(cls.boot_options[0])
        else:
            cls.save_csv = True

        print("Parameters used to run the bootstrap analysis:\n")
        print("         number of repeats = {:<10}".format(cls.n_repeats))
        print("      sampled trajectories = {:<10}".format(cls.n_samples))
        print("       confidence interval = {:<1}%".format(cls.ci_level))
        print(" ")

    @classmethod
    def run(cls, **kw):
        cls._load_params(**kw)
        cls._load_properties()
        cls._set_boot_params()

        print("Running the bootstrap algorithm...\n")
        df_bootstrap = bootstrap(
            cls.df_props,
            n_samples=cls.n_samples,
            n_repeats=cls.n_repeats,
            save_csv=cls.save_csv,
        )

        print("Creating statistics for the bootstrapped data...\n")
        df_stats = create_bootstrap_stats(df_bootstrap, cls.ci_level)
        df_stats.to_csv("stats_bootstrap.csv", index=False)
