"""chips_utils.py: Module is used to implement utility functions"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use(["science", "ieee"])

import os
import datetime as dt
import shutil
import numpy as np
from loguru import logger
from scipy import spatial

def date_to_get_carrington_rotation_number(d):
    """
    Convert the date to Carrington Rotation Number
    """
    unit = 27.2753
    ds = dt.datetime(1853,10,13) + dt.timedelta(0.6016)
    cycle = 0
    while ds < d:
        ds += dt.timedelta(unit)
        cycle += 1
    cycle -= 1
    logger.info(f"Carrington Rotation Number - {cycle}")
    return cycle

def download_riess21(fname="tmp/Reiss2021/riess21.zip",
                     url="https://figshare.com/ndownloader/articles/13397261/versions/2"):
    """
    Download all data from reomte to compare
    """
    os.makedirs("/".join(fname.split("/")[:-1]), exist_ok=True)
    if not os.path.exists(fname): 
        os.system(f"wget -O {fname} {url}")
        shutil.unpack_archive(fname, "/".join(fname.split("/")[:-1]))
    return

def get_data_by_model(model="ASSA", plot=False):
    """
    Get 2D array data by model name
    """
    o = np.empty((4096, 4096))
    fname = f"tmp/Reiss2021/binary{model}.txt"
    download_riess21()
    if os.path.exists(fname):
        o = np.loadtxt(fname)
        logger.info(f" Load - {fname}, [{o.shape}]")
        if plot: plot_saved_data("tmp/2018-05-30-12-00/", o, model)
    else: logger.warning(f" File - {fname}, does not exists.")
    return o

def plot_saved_data(folder, o, model):
    """
    Plot modeled data from Riess 2021
    """
    file = folder + f"{model}_analysis.png"
    k = 0
    fig, ax = plt.subplots(dpi=120, figsize=(3,3), nrows=1, ncols=1, sharex="all", sharey="all")
    cmap = matplotlib.cm.gray
    cmap.set_bad("k",1.)
    ax.imshow(o, origin="lower", cmap=cmap)
    ax.set_yticks([])
    ax.set_xticks([])
    fig.savefig(file, bbox_inches="tight")
    return

def plot_compare_data(folder, o, c, model, lim, sim=None, th=None):
    """
    Plot modeled data from Riess 2021 and CHIPS
    """
    file = folder + f"{model}_chips_{lim}_comp.png"
    fig, axes = plt.subplots(dpi=150, figsize=(6,3), nrows=1, ncols=2, sharex="all", sharey="all")
    cmap = matplotlib.cm.gray
    cmap.set_bad("k",1.)
    ax = axes[0]
    ax.imshow(o, origin="lower", cmap=cmap)
    ax.text(0.01, 1.05, "CHIPS", ha="left", va="center", transform=ax.transAxes)
    if th is not None: ax.text(0.99, 1.05, r"%s"%th, ha="right", va="center", transform=ax.transAxes)
    ax.set_yticks([])
    ax.set_xticks([])
    ax = axes[1]
    ax.imshow(c, origin="lower", cmap=cmap)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(0.01, 1.05, model, ha="left", va="center", transform=ax.transAxes)
    if sim is not None: ax.text(0.99, 1.05, r"%s"%sim, ha="right", va="center", transform=ax.transAxes)
    fig.savefig(file, bbox_inches="tight")
    return

def measure_similaity(im_x, im_y):
    """
    Measure similarity between images x and y
    """
    sim = {}
    #sim["cos"] = cosine_similarity(im_x.reshape(1, -1), im_y.reshape(1, -1))[0,0]
    sim["cos"] = 1 - spatial.distance.cosine(im_x.reshape(1, -1), im_y.reshape(1, -1))
    return sim


class Summary(object):
    """
    This class is dedicated for summary plots
    """
    
    def __init__(self, chips, b_dir):
        self.chips = chips
        self.b_dir = b_dir
        return
    
    def draw_axes(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    def create_mask_filter_4by4_plot(self):
        fname = self.b_dir + "analysis_01.png"
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(221)
        self.chips.aia.m_normalized.plot(annotate=False, axes=ax, vmin=self.chips.vmin)
        self.chips.aia.m_normalized.draw_limb()
        self.draw_axes(ax)
        ax = fig.add_subplot(222)
        ax.imshow(self.chips.sol_properties.sol_mask.i_mask, origin="lower", cmap="gray")
        self.draw_axes(ax)
        ax = fig.add_subplot(223)
        sol_disk = np.copy(self.chips.sol_properties.sol_filter.disk)
        sol_disk[sol_disk <= 0.] = 1
        sol_disk = np.log10(sol_disk)
        ax.imshow(sol_disk, origin="lower", cmap="gray", vmin=1, vmax=3)
        self.draw_axes(ax)
        ax = fig.add_subplot(224)
        sol_disk_filt = np.copy(self.chips.sol_properties.sol_filter.filt_disk)
        sol_disk_filt[sol_disk_filt <= 0.] = 1
        sol_disk_filt = np.log10(sol_disk_filt)
        ax.imshow(sol_disk_filt, origin="lower", cmap="gray", vmin=1, vmax=3)
        self.draw_axes(ax)
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        fig.savefig(fname, bbox_inches="tight")
        return
    
    def create_histogram_1by1_plot(self):
        fname = self.b_dir + "analysis_02.png"
        fig = plt.figure(figsize=(3,3), dpi=150)
        ax = fig.add_subplot(111)
        data, h_bins = self.chips.sol_properties.sol_histogram.data,\
                self.chips.sol_properties.sol_histogram.h_bins
        h, be, _ = ax.hist(data, bins=h_bins, histtype="step", color="r", density=True)
        peaks = self.chips.sol_properties.sol_histogram.peaks
        if len(peaks) > 0: ax.axvline(peaks[0], color="b", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"Intensity, $I_s$")
        ax.set_ylabel(r"Density, $D(I_s)$")
        fig.savefig(fname, bbox_inches="tight")
        return
    
    def create_histogram_NbyM_plot(self, slide=False):
        fname = self.b_dir + "analysis_05.png" if slide else self.b_dir + "analysis_04.png"
        xsplit, ysplit = self.chips.xsplit, self.chips.xsplit
        keys = self.chips.sol_properties.sol_histograms.keys
        regions = self.chips.sol_properties.sol_histograms.reg
        if slide: 
            keys = self.chips.sol_properties.sol_sld_histograms.keys
            regions = self.chips.sol_properties.sol_sld_histograms.reg
            xsplit, ysplit = int(np.sqrt(len(keys))), int(np.sqrt(len(keys)))
        fig, axes = plt.subplots(figsize=(3*xsplit,3*xsplit), dpi=150, nrows=xsplit, ncols=ysplit)
        k = 1
        for x in range(xsplit):
            for y in range(ysplit):
                reg = regions[keys[k-1]]
                ax = axes[x, y]
                data, h_bins = reg.data, reg.h_bins
                h, be, _ = ax.hist(data, bins=h_bins, histtype="step", color="r", density=True)
                peaks = reg.peaks
                if len(peaks) > 0: ax.axvline(peaks[0], color="b", ls="--", lw=0.8)
                ax.set_xscale("log")
                ax.set_xlabel(r"Intensity, $I_s$")
                ax.set_ylabel(r"Density, $D(I_s)$")
                k += 1
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        fig.savefig(fname, bbox_inches="tight")
        return
    
    def create_CH_CHB_plots(self):
        file = self.b_dir + "analysis_03.png"
        k = 0
        fig, axes = plt.subplots(dpi=120, figsize=(9,9), nrows=4, ncols=4, sharex="all", sharey="all")
        keys = list(self.chips.sol_properties.sol_chs_chbs.keys)
        pmap = self.chips.sol_properties.sol_chs_chbs.pmap
        dmap = self.chips.dmap
        cmap = matplotlib.cm.gray
        cmap.set_bad("k",1.)
        for i in range(4):
            for j in range(4):
                ax = axes[i,j]
                ax.imshow(vars(dmap)[str(keys[k])], origin="lower", cmap=cmap)
                self.draw_axes(ax)
                k += 2
                txt = r"$\theta=%.4f$"%vars(pmap)[str(keys[k])] + "\n" + "$I_{th}=%d$"%keys[k]
                ax.text(0.8, 0.8, txt, ha="center", va="center", transform=ax.transAxes,
                        fontdict={"size":7, "color":"w"})
                logger.warning(f" Limits - {keys[k]}")
        fig.subplots_adjust(wspace=0.01, hspace=0.05)
        fig.savefig(file, bbox_inches="tight")
        return
    
class SynSummary(object):
    
    def __init__(self, syn, b_dir):
        self.syn = syn
        self.b_dir = b_dir
        return
    
    def draw_axes(self, ax):
        ax.set_ylabel(r"Latitude [$^\circ$]")
        ax.set_xlabel(r"Carrington Longitude [$^\circ$]")
        ax.set_xticks(np.arange(0,361,90))
        ax.set_yticks(np.arange(-90,91,30))
        ax.set_xticklabels([r"%d$^\circ$"%x for x in np.arange(0,361,90)])
        ax.set_yticklabels([r"%d$^\circ$"%x for x in np.arange(-90,91,30)])
        return
    
    def create_raw_filter_plot(self):
        fname = self.b_dir + "analysis_01.png"
        fig = plt.figure(figsize=(4,4), dpi=180)
        ax = fig.add_subplot(211)
        data = np.ma.masked_invalid(self.syn.data)
        self.draw_axes(ax)
        ax.imshow(np.log10(data), origin="lower", cmap="sdoaia193", extent=[0, 360, -90, 90])
        txt = r"Synoptic Chart: $\lambda=%d\AA$"%self.syn.wavelength
        ax.text(0.01, 1.05, txt, ha="left", va="center", transform=ax.transAxes)
        txt = r"CR \#%d"%self.syn.cr_cycle
        ax.text(0.99, 1.05, txt, ha="right", va="center", transform=ax.transAxes)
        ax = fig.add_subplot(212)
        self.draw_axes(ax)
        ax.text(0.99, 1.05, r"Filtered by $%d\times %d$ GF"%(self.syn.medfilt_kernel, 
                                                              self.syn.medfilt_kernel),
                ha="right", va="center", transform=ax.transAxes)
        data = np.ma.masked_invalid(self.syn.syn_properties.syn_filter)
        ax.imshow(np.log10(data), origin="lower", cmap="sdoaia193", extent=[0, 360, -90, 90])
        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        fig.savefig(fname, bbox_inches="tight")
        return
    
    def create_histogram_1by1_plot(self):
        fname = self.b_dir + "analysis_02.png"
        fig = plt.figure(figsize=(3,3), dpi=150)
        ax = fig.add_subplot(111)
        data, h_bins = self.syn.syn_properties.syn_histogram.data,\
                self.syn.syn_properties.syn_histogram.h_bins
        h, be, _ = ax.hist(data, bins=h_bins, histtype="step", color="r", density=True)
        peaks = self.syn.syn_properties.syn_histogram.peaks
        if len(peaks) > 0: ax.axvline(peaks[0], color="b", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"Intensity, $I_s$")
        ax.set_ylabel(r"Density, $D(I_s)$")
        fig.savefig(fname, bbox_inches="tight")
        return
    
    def remove_axes(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    def create_CH_CHB_plots(self):
        file = self.b_dir + "analysis_03.png"
        k = 0
        fig, axes = plt.subplots(dpi=180, figsize=(18,9), nrows=4, ncols=4, sharex="all", sharey="all")
        keys = list(self.syn.syn_properties.syn_chs_chbs.keys)
        pmap = self.syn.syn_properties.syn_chs_chbs.pmap
        dmap = self.syn.dmap
        cmap = matplotlib.cm.gray
        cmap.set_bad("k",1.)
        for i in range(4):
            for j in range(4):
                ax = axes[i,j]
                ax.imshow(vars(dmap)[str(keys[k])], origin="lower", cmap=cmap, extent=[0, 360, -90, 90])
                self.draw_axes(ax)
                k += 2
                txt = r"$\theta=%.4f$"%vars(pmap)[str(keys[k])] + "\n" + "$I_{th}=%d$"%keys[k]
                ax.text(0.01, 1.02, txt, ha="left", va="bottom", transform=ax.transAxes,
                        fontdict={"size":7})
                logger.warning(f" Limits - {keys[k]}")
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        fig.savefig(file, bbox_inches="tight")
        return
    
    
if __name__ == "__main__":
    download_riess21()
    get_synoptic_fits_data(dt.datetime(2018,5,30,12), 193)