"""
    test_ROI_plots.py: Module is used to test the ROI plots
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
import copy
import os
import pickle

import numpy as np
from chips.chips import Chips
from chips.fetch import RegisterAIA
from chips.plots import Annotation, ImagePalette
import matplotlib.pyplot as plt


def run_CHIPS(param, indx):
    fname = f"tmp/stored_{indx}.pickle"
    print(fname)
    if not os.path.exists(fname):
        aia = RegisterAIA(
            date,
            [193],
            4096,
            apply_psf=False,
            local_file="sunpy/data/aia_lev1_{wavelength}a_{date_str}*.fits",
        )
        ch = Chips(
            aia, 
            medfilt_kernel=param["medfilt_kernel"], 
            h_bins=param["h_bins"], 
            threshold_range=[-3, 25]
        )
        ch.run_CHIPS(clear_prev_runs=True)
        with open(fname, "wb") as f:
            pickle.dump(
                dict(aia=ch.aia, param=param), f
            )
    else:
        with open(fname, "rb") as f:
            o = pickle.load(f)
        ch = Chips(
                o["aia"], 
                medfilt_kernel=param["medfilt_kernel"], 
                h_bins=param["h_bins"], 
                threshold_range=[-3, 25]
            )
    return ch

###################################################################
# This code snippet is to show how to generate ROI plots [Fig 4]
# Steps to generate the example plots
#   1. Run the CHIPS algorithms with different parameters
#   2. Create reference plots
#   3. Create ROI plot just like Fig 4 in 
###################################################################
def run_different_CHIPS_parameters(date, base_number=0):
    """
    Run the CHIPS algorithms with different parameters 
    and save them for future plots.
    """
    cps = []
    params = [
        dict(medfilt_kernel=11, h_bins=5000),
        dict(medfilt_kernel=31, h_bins=5000),
        dict(medfilt_kernel=51, h_bins=5000),
        dict(medfilt_kernel=71, h_bins=5000),
        dict(medfilt_kernel=51, h_bins=20000),
        dict(medfilt_kernel=51, h_bins=10000),
        dict(medfilt_kernel=51, h_bins=5000),
        dict(medfilt_kernel=51, h_bins=500),
    ]
    #####################################################################
    # Run this part one at a time to generate stable save files
    for i in range(base_number, base_number+4):
        d = run_CHIPS(params[i], i)
        cps.append(copy.copy(d))
        del d
    #####################################################################  
    return cps, params

def draw_ref_plots():
    """
    """
    return

def draw_ROI_plot(date, cps, params, base_number=0):
    """
    Draw a panel of 2X2 plots for ROI plots.
    """
    ip = ImagePalette(
        figsize=(6, 6),
        dpi=300,
        nrows=2,
        ncols=2,
        vert=np.array([[1300,1300],[2500,2500]])
    )
    dx = cps[0].aia.datasets[193][4096]

    def plot_roi(index=0, cb=False, a_off=True):
        d = cps[index].aia.datasets[193][4096]
        ip.draw_colored_disk(
            map=d.normalized,
            pixel_radius=d.pixel_radius,
            resolution=d.resolution,
            axis_off=a_off,
        )
        ip.ovearlay_localized_regions(
            d.solar_ch_regions, 
            prob_lower_lim=0,
            add_color_bar=cb,
            axis_off=a_off
        )
        return plt.gcf().axes
    
    plot_roi(0)
    plot_roi(1)
    plot_roi(2, a_off=False)
    plot_roi(3, cb=True)
    ax = plt.gcf().axes[2]
    ax.set_xticks(np.arange(1300, 2501, 400))
    ax.set_yticks(np.arange(1300, 2501, 400))
    ax.set_xticklabels(np.arange(1300, 2501, 400))
    ax.set_yticklabels(np.arange(1300, 2501, 400))
    ax.set_ylabel("pixels")
    ax.set_xlabel("pixels")

    annotations = []
    annotations.append(
        Annotation(
            date.strftime("%Y-%m-%d %H:%M"),
            1.05,
            0.99,
            "center",
            "top",
            rotation=90,
        )
    )
    annotations.append(
        Annotation(
            r"$\lambda=%d\AA$" % dx.wavelength,
            0.95,
            1.05,
            "right",
            "center",
        )
    )
    ip.annotate(annotations, ticker=1,)
    ip.annotate(
        [Annotation(
            r"$\kappa=$%d, $h_{bins}=$%d" % 
            (params[0+base_number]["medfilt_kernel"], params[0+base_number]["h_bins"]),
            0.05,
            0.9,
            "left",
            "center",
        )], 
        ticker=0,
    )
    ip.annotate(
        [Annotation(
            r"$\kappa=$%d, $h_{bins}=$%d" % 
            (params[1+base_number]["medfilt_kernel"], params[1+base_number]["h_bins"]),
            0.05,
            0.9,
            "left",
            "center",
        )], 
        ticker=1,
    )
    ip.annotate(
        [Annotation(
            r"$\kappa=$%d, $h_{bins}=$%d" % 
            (params[2+base_number]["medfilt_kernel"], params[2+base_number]["h_bins"]),
            0.05,
            0.9,
            "left",
            "center",
        )], 
        ticker=2,
    )
    ip.annotate(
        [Annotation(
            r"$\kappa=$%d, $h_{bins}=$%d" % 
            (params[3+base_number]["medfilt_kernel"], params[3+base_number]["h_bins"]),
            0.05,
            0.9,
            "left",
            "center",
        )], 
        ticker=3,
    )
    
    ip.save(f"tmp/ROI.png")
    ip.close()
    return


if __name__ == "__main__":
    date, base_number = dt.datetime(2015, 8, 20), 4
    cps, params = run_different_CHIPS_parameters(date, base_number=base_number)
    draw_ROI_plot(date, cps, params, base_number=base_number)