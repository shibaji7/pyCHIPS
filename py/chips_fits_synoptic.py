"""chips_fits.py: Module is used to implement edge detection tecqniues using thresholding"""

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
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm

import os
import datetime as dt
import numpy as np
from scipy import signal, stats
import cv2
import json
from loguru import logger
import pickle
from argparse import Namespace

from astropy.utils.data import download_file
import sunpy.map
import sunpy.io

import chips_utils as cutils

class Synoptic(object):
    
    def __init__(self, args):
        """
        Get the Synoptic data
        """
        for k in vars(args).keys():
            setattr(self, k, vars(args)[k])
        self.cr_cycle = cutils.date_to_get_carrington_rotation_number(self.date)
        self.files.dir = self.files.synoptic.format(CR=self.cr_cycle)
        # Create / clear base directory
        os.makedirs(self.files.dir, exist_ok=True)
        if self.clear: self.clear_last_run()
        # Read FITS data using sunpy
        self.load_fits()
        # Load basic parameters
        self.extract_basic_properties()
        # Run median filter
        self.run_median_filters()
        # Extract histogram
        self.extract_histogram()
        # Extract CH CHBs
        self.extract_CHs_CHBs()
        self.save_summary_plots()
        return
    
    def load_fits(self):
        """
        Load FITS data
        """
        sdo_file_location = "https://sdo.gsfc.nasa.gov/assets/img/synoptic/AIA%04d/CR%d.fits"
        file = sdo_file_location%(self.wavelength, self.cr_cycle)
        logger.info(f"Download- {file}")
        filename = download_file(file, cache=True)
        self.data, self.header = sunpy.io.fits.read(filename)[0] 
        self.header["cunit1"] = "arcsec"
        self.header["cunit2"] = "arcsec"
        return
    
    def extract_basic_properties(self):
        logger.info("Extract solar basic properties.")
        self.syn_properties = Namespace()
        for h in self.header:
            if isinstance(self.header[h], dict): setattr(self.syn_properties, h, 
                                                         Namespace(**self.header[h]))
            else: setattr(self.syn_properties, h, self.header[h])
        return
    
    def run_median_filters(self):
        self.load_last_run()
        if not hasattr(self.syn_properties, "syn_filter"):
            logger.info("Run Median filters.")
            syn = np.ma.masked_invalid(self.data)
            self.syn_filt = signal.medfilt2d(syn, self.medfilt_kernel)
            self.syn_properties.syn_filter = self.syn_filt
            self.save_current_run()
        logger.info("Done ssynoptic median filter.")
        return
    
    def extract_histogram(self):
        if not hasattr(self.syn_properties, "syn_histogram"):
            logger.info("Extract synoptic histograms.")
            h_data = self.syn_properties.syn_filter
            d = h_data.ravel()[~np.isnan(h_data.ravel())]
            h, be = np.histogram(d, bins=self.h_bins, density=True)
            if self.h_thresh is None: self.h_thresh = np.max(h)/self.ht_peak_ratio
            peaks, _ = signal.find_peaks(h, height=self.h_thresh)
            bc = be[:-1] + np.diff(be)
            self.peaks = bc[peaks]
            self.syn_properties.syn_histogram = Namespace(**dict(
                peaks = self.peaks, hbase = h, bound = bc, data = d, h_thresh=self.h_thresh,
                h_bins=self.h_bins
            ))
            self.save_current_run()
        logger.info("Done extracting histograms.")
        return
    
    def extract_CHs_CHBs(self):
        scale = 5
        threshold_range = np.arange(self.threshold_range[0], self.threshold_range[1])
        if not hasattr(self.syn_properties, "syn_chs_chbs"):
            logger.info("Extract CHs and CHBs.")
            if len(self.syn_properties.syn_histogram.peaks) > 0:
                l0 = self.syn_properties.syn_histogram.peaks[0]
                limits = np.round(l0 + threshold_range, 4)
                data = self.syn_properties.syn_filter
                self.im_shape = (int(data.shape[0]/scale), int(data.shape[1]/scale))
                dtmp_map, prob_map, lth_map = {}, {}, {}
                for lim in limits:
                    lth_map[str(lim)] = lim
                    dtmp = np.copy(data)
                    dtmp[dtmp <= lim] = -1
                    dtmp[dtmp > lim] = 0
                    dtmp[dtmp == -1] = 1
                    dtmp[np.isnan(dtmp)] = 0
                    dtmp_map[str(lim)] = cv2.resize(np.copy(dtmp), self.im_shape)
                    dm = np.copy(data).ravel()
                    p = self.calculate_prob(dm, lim, l0)
                    prob_map[str(lim)] = p
                    logger.info(f"Estimated prob.({p}) at lim({lim})")
                self.syn_properties.syn_chs_chbs = Namespace(**dict( 
                    pmap = Namespace(**prob_map), 
                    lmap = Namespace(**lth_map),
                    keys = limits
                ))
                self.dmap = Namespace(**dtmp_map)
                self.save_current_run()
            else: logger.warning("No threhold peak detected!")
        else:
            limits = self.syn_properties.syn_chs_chbs.keys
            dtmp_map = {}
            data = self.syn_properties.syn_filter
            self.im_shape = (int(data.shape[0]/scale), int(data.shape[1]/scale))
            for lim in limits:
                dtmp = np.copy(data)
                dtmp[dtmp <= lim] = -1
                dtmp[dtmp > lim] = 0
                dtmp[dtmp == -1] = 1
                dtmp[np.isnan(dtmp)] = 0
                dtmp_map[str(lim)] = np.copy(dtmp)#cv2.resize(np.copy(dtmp), self.im_shape)
            self.dmap = Namespace(**dtmp_map)
        logger.info("Done with extract_CHs_CHBs!")
        return
    
    def calculate_prob(self, dm, lth, l0):
        """
        Estimate probability for each map
        """
        dm = dm[~np.isnan(dm)]
        dm = dm[dm <= lth]
        ps = 1 / ( 1 + np.exp(dm-l0).round(4) )
        h, e = np.histogram(ps,bins=np.linspace(0,1,21))
        b = (e + np.diff(e)[0]/2)[:-1]
        p = np.sum(b[b>.5]*h[b>.5])/np.sum(b*h)
        return np.round(p,4)
    
    def save_summary_plots(self):
        """
        Save all summary plot for synoptic analysis
        """
        summ = cutils.SynSummary(self, self.files.dir)
        summ.create_raw_filter_plot()
        summ.create_histogram_1by1_plot()
        if hasattr(self, "dmap"): summ.create_CH_CHB_plots()
        return
    
    def save_current_run(self):
        """
        Save run to a .pickle file
        """
        fname = self.files.dir + "syn_properties.pickle"
        with open(fname, "wb") as h: pickle.dump(self.syn_properties, h, protocol=pickle.HIGHEST_PROTOCOL)
        return
    
    def load_last_run(self):
        """
        Load latest run to sol_properties
        """
        fname = self.files.dir + "syn_properties.pickle"
        if os.path.exists(fname):
            with open(fname, "rb") as h: self.syn_properties = pickle.load(h)
        else: self.save_current_run()
        return
    
    def clear_last_run(self):
        """
        Clear latest run
        """
        os.system(f"rm -rf {self.files.dir[:-1]}/*")
        return