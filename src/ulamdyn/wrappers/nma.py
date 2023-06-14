# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: June 14, 2023

from ulamdyn.nma.normal_mode_analysis import NormalModeAnalysis

__all__ = ["NMAnalysis"]


class NMAnalysis:
    @classmethod
    def _load_params(cls, **kw):
        # List of valid keywords
        keywords = [
            "vib_file",
            "time_intervals",
        ]
        for k in keywords:
            val = kw.get(k)
            setattr(cls, k, val)

        if cls.time_intervals != "":
            cls.time_intervals = cls.time_intervals.strip().split(",")
        else:
            cls.time_intervals = []

    @classmethod
    def run(cls, **kw):
        cls._load_params(**kw)
        nma_obj = NormalModeAnalysis()
        nma_obj = nma_obj.run(cls.vib_file)

        if len(cls.time_intervals) >= 1:
            nma_obj = nma_obj.make_stats(time_intervals=cls.time_intervals)

        nma_obj.save_csv()
