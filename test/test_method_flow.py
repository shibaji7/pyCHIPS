"""
    test_method_flow.py: Module is used to test method flows and generate the plots
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

#################################################
# Method runs and stores all possible parameters
#################################################
def run_CHIPS(date, param):
    fname = f"tmp/test_method_data_{param['wavelength']}.pickle"
    if not os.path.exists(fname):
        aia = RegisterAIA(
            date,
            [param["wavelength"]],
            4096,
            apply_psf=False,
            local_file="sunpy/data/aia_lev1_{wavelength}a_{date_str}*.fits",
        )
        ch = Chips(
            aia, 
            medfilt_kernel=param["medfilt_kernel"], 
            h_bins=param["h_bins"],
            threshold_range=param["threshold_range"],
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
            threshold_range=param["threshold_range"],
        )
    return ch

def __create_localized_maps__(disk, regions, ip, id, axis_off=True):
    keys = list(regions.__dict__.keys())
    limits, probs = (
        np.array([regions.__dict__[key].lim for key in keys]),
        np.array([regions.__dict__[key].prob for key in keys]),
    )
    n_probs = (probs - probs.min()) / (probs.max() - probs.min())
    p, key = n_probs[id], keys[id]
    txt = r"$I_{th}=$%s" % key + "\n" + r"$\theta=%.3f$" % p
    map = regions.__dict__[key].map
    ip.plot_binary_localized_map(
        map, disk.pixel_radius, disk.resolution, 
        None, txt, axis_off=axis_off
    )
    return

def draw_flow_example_plot(date, chips, param):
    ip = ImagePalette(
        figsize=(9, 9),
        dpi=300,
        nrows=3,
        ncols=3,
        sharex="none",
        sharey="none",
    )
    d = chips.aia.datasets[param["wavelength"]][4096]
    ip.draw_colored_disk(
        map=d.raw,
        pixel_radius=d.pixel_radius,
        resolution=d.resolution,
    )
    ip.draw_grayscale_disk(
        map=d.normalized,
        pixel_radius=d.pixel_radius,
        resolution=d.resolution,
    )
    ip.draw_grayscale_disk(
        map=d.normalized,
        data=d.solar_filter.filt_disk,
        pixel_radius=d.pixel_radius,
        resolution=d.resolution,
    )
    ax = ip.__axis__(axis_off=False)
    ax.tick_params(
        axis='both', which='both', bottom=False, top=False, 
        left=False, right=False, labelbottom=False, labelleft=False
    )
    ax.step(d.histogram.bound, d.histogram.hbase, 
            where="mid", ls="-", lw=1, color="b")
    if param["wavelength"] == 171:
        xlim, bounds = [0, 50], [7, 3, 10]
    if param["wavelength"] == 193:
        xlim, bounds = [0, 200], [5, 3, 10]
    if param["wavelength"] == 211:
        xlim, bounds = [0, 50], [7, 3, 10]
    ax.set_xlim(xlim)
    ax.axvline(d.histogram.bound[bounds[0]], color="r", ls="--", lw=0.8)
    ax.axvspan(d.histogram.bound[bounds[1]], d.histogram.bound[bounds[2]], color="r", alpha=0.2)
    __create_localized_maps__(d, d.solar_ch_regions, ip, 0)
    __create_localized_maps__(d, d.solar_ch_regions, ip, 2)
    __create_localized_maps__(d, d.solar_ch_regions, ip, 4, axis_off=False)
    __create_localized_maps__(d, d.solar_ch_regions, ip, 6)
    ip.draw_colored_disk(
        map=d.raw,
        pixel_radius=d.pixel_radius,
        resolution=d.resolution,
    )
    ip.ovearlay_localized_regions(d.solar_ch_regions, prob_lower_lim=0)

    for i, t in enumerate(["a", "b", "c", "d", "e", "f", "g", "h", "i"]):
        ip.annotate(
            [
                Annotation(
                    t, 0.95, 0.95, "center", "center",
                    fontdict={"color": "w", "size": 12},
                )
            ], ticker=i
        )
    ip.annotate(
        [
            Annotation(
                "d", 0.95, 0.95, "center", "center",
                fontdict={"color": "k", "size": 12},
            )
        ], ticker=3
    )
    ip.annotate(
        [
            Annotation(
                date.strftime("%Y-%m-%d %H:%M"), -0.1, 0.95, "left", "top",
                rotation=90, fontdict={"color": "k", "size": 12},
            ),
            Annotation(
                r"$\lambda=$%d$\AA$"%param["wavelength"], 0.05, 1.05, "left", "center",
                fontdict={"color": "k", "size": 12},
            )
        ], ticker=0
    )
    ip.annotate(
        [
            Annotation(
                r"$h_{bins}=$%d"%param["h_bins"], 0.05, 1.05, "left", "center",
                fontdict={"color": "k", "size": 12},
            ),
            Annotation(
                r"$\kappa=$%d"%param["medfilt_kernel"], 0.95, 1.05, "right", "center",
                fontdict={"color": "k", "size": 12},
            )
        ], ticker=1
    )
    ip.save(f"tmp/flow_steps.png")
    ip.close()
    return

if __name__ == "__main__":
    date = dt.datetime(2015, 8, 20)
    # Params for 193 Dataset
    param = dict(
        medfilt_kernel=51, h_bins=500,
        wavelength=193, threshold_range=[0, 20]
    )
    # Params for 211 Dataset
    # param = dict(
    #     medfilt_kernel=51, h_bins=1000,
    #     wavelength=211, threshold_range=[-3, 11]
    # )
    
    chips = run_CHIPS(date, param)
    draw_flow_example_plot(date, chips, param)