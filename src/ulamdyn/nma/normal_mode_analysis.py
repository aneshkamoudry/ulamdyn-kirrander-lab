# author: Felix Plasser and Max Pinheiro Jr <maxjr82@gmail.com>
#   Date: June 2, 2023
#  Descr: Module used to perform normal mode analysis.
#         Coordinates are transformed into the normal mode basis. As a first step,
#         each geometry from MD should be aligned to a reference structure.

import numpy as np
import pandas as pd

from ulamdyn.data_loader import GetCoords
from ulamdyn.statistics import aggregate_data
from ulamdyn.nma.vib_molden import VibMolden

__all__ = ["NormalModeAnalysis"]


class NormalModeAnalysis(GetCoords):
    def __str__(self) -> str:
        """Provide a string representation of the class.

        :return: Short description of the class functionality.
        :rtype: str
        """
        return "Project MD coordinates into the normal mode basis."

    def __init__(self) -> None:
        """Class constructor."""
        super().__init__()
        self.read_eq_geom()
        self.read_all_trajs()

        self.nm_basis = None
        self.dataset = None
        self.vib_data = None
        self.stats_data = {}
        self._last_called = None

    def _build_nm_basis(self, molden_inp: str) -> None:
        self.vmol = VibMolden()
        self.vmol.read_molden_file(molden_inp)
        vib_mat = self.vmol.get_vib_matrix()
        self.nm_basis = np.linalg.inv(vib_mat)

    def _calc_projections(self) -> np.ndarray:
        all_geoms = self.xyz.copy()
        ref_geom = self.eq_xyz.copy()
        n_samples, n_atoms, _ = all_geoms.shape
        nm_matrix = np.empty((n_samples, 3 * n_atoms), dtype=np.float64)

        for i, xyz in enumerate(all_geoms):
            geom_diff = xyz.flatten() - ref_geom.flatten()
            nm_vec = np.dot(geom_diff, self.nm_basis)
            nm_matrix[i] = nm_vec

        return nm_matrix

    def save_csv(self) -> None:
        """Save the datasets into a csv file"""
        if self._last_called == "run":
            df = self.dataset
            df.to_csv("all_nma.csv", index=False, header=True)
            df_vib = self.vib_data
            df_vib.to_csv("vib_ref_geom.csv", index=False, header=True)
        if self._last_called == "make_stats":
            for k in self.stats_data:
                filename = k + ".csv"
                df = self.stats_data[k]
                df.to_csv(filename, index=False, header=True)

    def run(self, molden_ref: str = "freq.molden"):
        """Perform the Normal Mode analysis for MD geometries."""
        # STEP 1: Create the Normal Mode basis from a reference molden file containing
        #         the vibrational frequencies
        self._build_nm_basis(molden_ref)
        # STEP 2: Project the difference vector between the geometry of each time step
        #         and the reference one into the normal mode basis
        nm_matrix = self._calc_projections()
        # STEP 3: Build dataframe with the resulting Normal Mode matrix for all samples
        n_modes = self.nm_basis.shape[0]
        col_names = ["NM" + str(i) for i in range(1, n_modes + 1)]
        df_nma = pd.DataFrame(nm_matrix, columns=col_names)
        df_nma.insert(0, "TRAJ", self.traj_time[:, 0])
        df_nma.insert(1, "time", self.traj_time[:, 1])
        df_nma["TRAJ"] = df_nma["TRAJ"].astype("int32")
        df_nma["RMSD"] = self.rmsd
        self.dataset = df_nma
        # STEP 4: Create a dataframe with the frequencies (cm-1) and period (fs) for
        #         each normal mode of the reference structure
        ref_vib_data = self.vmol.get_vib_data()
        nm_labels = ref_vib_data[0]
        freqs = [np.float64(i) for i in ref_vib_data[1]]
        periods = [np.float64(i) for i in ref_vib_data[2]]
        ref_vib_data = {"Wavenumber (1/cm)": freqs, "Period (fs)": periods}
        df_vib_data = pd.DataFrame.from_dict(
            ref_vib_data, orient="index", columns=nm_labels
        )
        self.vib_data = df_vib_data
        self._last_called = "run"
        return self

    def make_stats(self, by: list = ["time"], time_intervals: list = []):
        if self.dataset is not None:
            # STEP 1: Calculate statistics (mean, std and more) over trajectories
            #         against the time.
            df_nma_stats = aggregate_data(self.dataset, by)
            if len(time_intervals) >= 1:
                dfs = []
                cols = ["time"] + [c for c in df_nma_stats.columns if "mean" in c]
                df_mean = df_nma_stats[cols]
                for dt in time_intervals:
                    t_start, t_end = [float(i) for i in dt.split("-")]
                    t_slice = f"time >= {t_start} and time <= {t_end}"
                    df_temp = df_mean.query(t_slice)
                    df_temp = df_temp.drop("time", axis=1).std()
                    dfs.append(df_temp)
                df_std = pd.concat(dfs, axis=1)
                df_std.columns = time_intervals
                df_std = df_std.reset_index()
                self.stats_data["cross_avg_std"] = df_std
            self.stats_data["stats_nma"] = df_nma_stats
            self._last_called = "make_stats"
            return self
        else:
            print("The dataset variable is empty. Statistics cannot be performed.")
