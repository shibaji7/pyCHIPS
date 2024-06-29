"""
    test_simulations.py: Module is used to test chips simulations.
"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import datetime as dt
import numpy as np
import unittest

from chips.fetch import RegisterAIA
from chips.plots import Annotation, ImagePalette
from chips.chips import Chips
# import sys
# sys.path.append("../chips/")
# from fetch import RegisterAIA

def plot_histograms(hist0):
    import scipy.stats as stats
    fit_alpha, fit_loc, fit_beta=stats.gamma.fit(hist0)
    x = np.arange(1, 10000)
    pdf = stats.gamma.pdf(x, fit_alpha, fit_loc, fit_beta)
    ip = ImagePalette(
            figsize=(9, 3),
            dpi=300,
            nrows=1,
            ncols=3,
        )
    ax = ip.__axis__(axis_off=False)
    ax.hist(hist0, bins=1000, histtype="step", color="r", density=True)
    
    #ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"$I_s$ from .fits files")
    ax.set_ylabel(r"Occurance Rate/Density")
    ax.set_xlim([1, 1000])
    ax.plot(x, pdf*1.5, "k-", lw=0.8)
    ip.save("tmp/hisograms.png")
    ip.close()
    return


class TestCHIPS(unittest.TestCase):
    def test_fetch_193_solar_disk_(self):
        aia = RegisterAIA(dt.datetime(2018, 5, 30, 12), wavelengths=[193], apply_psf=False)
        #aia.plot_scatter_maps()
        chips = Chips(
            aia,
            medfilt_kernel=51,
            h_bins=500
        )
        chips.run_CHIPS()
        # disk = chips.aia.datasets[193][4096]
        # h_data = disk.solar_filter.filt_disk * disk.solar_mask.n_mask
        # h_data = h_data.ravel()[~np.isnan(h_data.ravel())]
        # plot_histograms(h_data)
        #chips.run_CHIPS(clear_prev_runs=True)
        #chips.to_netcdf(193, 4096)
        # disk = chips.aia.datasets[193][4096]
        # map0, map = (
        #     getattr(disk.solar_ch_regions, str(18.2434)).map,
        #     getattr(disk.solar_ch_regions, str(19.2434)).map
        # )
        # measures = chips.compute_similarity_measures(map, map0)
        return


if __name__ == "__main__":
    unittest.main()
