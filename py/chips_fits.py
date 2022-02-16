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

import astropy.units as u
from sunpy.net import Fido, attrs
import sunpy.map
from aiapy.calibrate import register, update_pointing, normalize_exposure
import aiapy.psf

from astropy.utils.data import download_file
import sunpy.map
import sunpy.io

import chips_utils as cutils

class RegisterAIA(object):
    
    def __init__(self, date, wavelength=193, resolution=4096, vmin=10, apply_psf=False):
        self.wavelength = wavelength
        self.date = date
        self.resolution = resolution
        self.vmin = vmin
        self.apply_psf = apply_psf
        self.normalized()
        return
    
    def normalized(self):
        logger.info(f"Normalize map using L({self.wavelength}), R({self.resolution}) on {self.date}")
        q = Fido.search(
            attrs.Time(self.date.strftime("%Y-%m-%dT%H:%M:%S"), (self.date + dt.timedelta(seconds=11)).strftime("%Y-%m-%dT%H:%M:%S")),
            attrs.Instrument("AIA"),
            attrs.Wavelength(wavemin=self.wavelength*u.angstrom, wavemax=self.wavelength*u.angstrom),
        )
        logger.info(f"Record found: len({len(q)})")
        logger.info(f"Registering FITS data")
        self.m = sunpy.map.Map(Fido.fetch(q[0,0]))
        if self.apply_psf:
            psf = aiapy.psf.psf(self.m.wavelength)
            self.m = aiapy.psf.deconvolve(self.m, psf=psf)
        m_updated_pointing = update_pointing(self.m)
        m_registered = register(m_updated_pointing)
        self.m_normalized = normalize_exposure(m_registered)
        return
    
class CHIPS(object):
    
    def __init__(self, args):
        """
        Get the AIA data
        """
        for k in vars(args).keys():
            setattr(self, k, vars(args)[k])
        self.files.dir = self.files.dir.format(date=self.date.strftime("%Y-%m-%d-%H-%M"))
        # Create / clear base directory
        os.makedirs(self.files.dir, exist_ok=True)
        if self.clear: self.clear_last_run()
        # Fetch raw fits data from webserver
        self.fetch_raw_fits_data()
        # Initialize sol_properties 
        self.extract_basic_properties()
        # Extract solar disk mask
        self.extract_solar_masks()
        # Run smoothing Gaussian filters
        self.run_filters()
        # Extract histogram - for whole map
        self.extract_histogram()
        # Extract histogram - for partial map
        self.extract_histograms()
        # Extract histogram - for partial map - sliding window
        self.extract_sliding_histograms()
        # Extract CH abd CHB
        self.extract_CHs_CHBs()
        # Run model validation
        self.run_validation()
        # Create summary plots
        self.save_summary_plots()
        return

    def fetch_raw_fits_data(self):
        """
        Fetch raw FITS data from webserver
        """
        self.aia = RegisterAIA(self.date, self.wavelength, self.resolution, self.vmin)
        self.fits_data = self.aia.m_normalized.data
        return
    
    def extract_basic_properties(self):
        """
        Fetch properties from the 
        """
        logger.info("Extract solar basic properties.")
        self.sol_properties = Namespace(**dict(
            sol_radius = Namespace(**dict(
                arcsec=self.aia.m_normalized.rsun_obs.value,
                pix=int(self.aia.m_normalized.rsun_obs.value/self.rscale)
            ))
        ))
        return
    
    def extract_solar_masks(self):
        """
        Extract solar masks
        """
        self.load_last_run()
        if not hasattr(self.sol_properties, "sol_mask"):
            logger.info("Extract solar mask.")
            self.mask = np.zeros_like(self.fits_data)*np.nan
            cv2.circle(self.mask, (int(self.resolution/2), int(self.resolution/2)), 
                       int(self.sol_properties.sol_radius.arcsec/self.rscale), 255, -1)
            self.mask[self.mask > 1] = 1.
            self.i_mask = np.copy(self.mask)
            self.i_mask[np.isnan(self.i_mask)] = 0.
            self.i_mask[self.i_mask > 1] = 1.
            self.sol_properties.sol_mask = Namespace(**dict(
                n_mask = self.mask,
                i_mask = self.i_mask
            ))
            self.save_current_run()
        logger.info("Done solar mask.")
        return
    
    def run_filters(self):
        if not hasattr(self.sol_properties, "sol_filter"):
            logger.info("Run Median filters.")
            self.sol_disk = self.i_mask * self.fits_data.data
            self.sol_disk_filt = self.i_mask * signal.medfilt2d(self.fits_data, self.medfilt_kernel)
            self.sol_properties.sol_filter = Namespace(**dict(
                disk = self.sol_disk, filt_disk = self.sol_disk_filt
            ))
            self.save_current_run()
        logger.info("Done mdeian filters.")
        return
    
    def extract_histogram(self):
        """
        Extract histograms for whole map
        """
        if not hasattr(self.sol_properties, "sol_histogram"):
            logger.info("Extract solar histograms.")
            h_data = self.sol_properties.sol_filter.filt_disk * self.sol_properties.sol_mask.n_mask
            d = h_data.ravel()[~np.isnan(h_data.ravel())]
            h, be = np.histogram(d, bins=self.h_bins, density=True)
            if self.h_thresh is None: self.h_thresh = np.max(h)/self.ht_peak_ratio
            peaks, _ = signal.find_peaks(h, height=self.h_thresh)
            bc = be[:-1] + np.diff(be)
            self.peaks = bc[peaks]
            self.sol_properties.sol_histogram = Namespace(**dict(
                peaks = self.peaks, hbase = h, bound = bc, data = d, h_thresh=self.h_thresh,
                h_bins=self.h_bins
            ))
            self.save_current_run()
        logger.info("Done extracting histograms.")
        return
    
    def extract_histograms(self):
        """
        Extract histograms for each section
        """
        if not hasattr(self.sol_properties, "sol_histograms")\
            and (self.xsplit is not None) and (self.ysplit is not None):
            regions, r_peaks = {}, []
            k = 0
            filt_disk, n_mask = self.sol_properties.sol_filter.filt_disk, self.sol_properties.sol_mask.n_mask
            xunit, yunit = int(self.resolution/self.xsplit), int(self.resolution/self.ysplit)
            for x in range(self.xsplit):
                for y in range(self.ysplit):
                    r_disk = filt_disk[x*xunit:(x+1)*xunit, y*yunit:(y+1)*xunit]
                    r_mask = n_mask[x*xunit:(x+1)*xunit, y*yunit:(y+1)*xunit]
                    d = (r_disk*r_mask).ravel()
                    d = d[~np.isnan(d.ravel())]
                    h, be = np.histogram(d, bins=self.h_bins, density=True)
                    if self.h_thresh is None: self.h_thresh = np.max(h)/self.ht_peak_ratio
                    peaks, _ = signal.find_peaks(h, height=self.h_thresh)
                    bc = be[:-1] + np.diff(be)
                    if len(peaks) > 1: r_peaks.extend(peaks[:2])
                    regions[k] = Namespace(**dict(
                        peaks = peaks, hbase = h, bound = bc, data = d, h_thresh=self.h_thresh,
                        h_bins = self.h_bins
                    ))
                    k += 1
            self.sol_properties.sol_histograms = Namespace(**dict(
                reg = regions, fpeak = r_peaks, keys=range(k)
            ))
            self.save_current_run()
        return
    
    def extract_sliding_histograms(self):
        """
        Extract histograms for each section
        """
        if not hasattr(self.sol_properties, "sol_sld_histograms")\
            and (self.xsplit is not None) and (self.ysplit is not None):
            regions, r_peaks = {}, []
            k = 0
            filt_disk, n_mask = self.sol_properties.sol_filter.filt_disk, self.sol_properties.sol_mask.n_mask
            for xs in range(0, self.resolution-self.sliding_winow.wlen, self.sliding_winow.wsep):
                for ys in range(0, self.resolution-self.sliding_winow.wlen, self.sliding_winow.wsep):
                    xl, yl = xs + self.sliding_winow.wlen, ys + self.sliding_winow.wlen
                    r_disk = filt_disk[xs:xl, ys:yl]
                    r_mask = n_mask[xs:xl, ys:yl]
                    d = (r_disk*r_mask).ravel()
                    d = d[~np.isnan(d.ravel())]
                    h, be = np.histogram(d, bins=self.h_bins, density=True)
                    if self.h_thresh is None: self.h_thresh = np.max(h)/self.ht_peak_ratio
                    peaks, _ = signal.find_peaks(h, height=self.h_thresh)
                    bc = be[:-1] + np.diff(be)
                    if len(peaks) > 1: r_peaks.extend(peaks[:2])
                    regions[k] = Namespace(**dict(
                        peaks = peaks, hbase = h, bound = bc, data = d, h_thresh=self.h_thresh,
                        h_bins = self.h_bins
                    ))
                    k += 1
            self.sol_properties.sol_sld_histograms = Namespace(**dict(
                reg = regions, fpeak = r_peaks, keys=range(k)
            ))
            self.save_current_run()
        return
    
    def extract_CHs_CHBs(self):
        """
        Find the CHs and CHBs and associated probability
        """
        threshold_range = np.arange(self.threshold_range[0], self.threshold_range[1])
        if not hasattr(self.sol_properties, "sol_chs_chbs"):
            logger.info("Extract CHs and CHBs.")
            if len(self.sol_properties.sol_histogram.peaks) > 0:
                l0 = self.sol_properties.sol_histogram.peaks[0]
                limits = np.round(l0 + threshold_range, 4)
                data = self.sol_properties.sol_filter.filt_disk * self.sol_properties.sol_mask.n_mask
                dtmp_map, prob_map, lth_map = {}, {}, {}
                for lim in limits:
                    lth_map[str(lim)] = lim
                    dtmp = np.copy(data)
                    n_mask = np.zeros_like(data) * np.nan
                    cv2.circle(n_mask, (int(self.resolution/2), int(self.resolution/2)), 
                               int(self.sol_properties.sol_radius.arcsec/self.rscale), 255, -1)
                    n_mask[n_mask > 0] = 1
                    dtmp = n_mask * dtmp
                    dtmp[dtmp <= lim] = -1
                    dtmp[dtmp > lim] = 0
                    dtmp[dtmp == -1] = 1
                    dtmp[np.isnan(dtmp)] = 0
                    dtmp_map[str(lim)] = cv2.resize(np.copy(dtmp), (1024, 1024))
                    dm = np.copy(data).ravel()
                    p = self.calculate_prob(dm, lim, l0)
                    prob_map[str(lim)] = p
                    logger.info(f"Estimated prob.({p}) at lim({lim})")
                self.sol_properties.sol_chs_chbs = Namespace(**dict( 
                    pmap = Namespace(**prob_map), 
                    lmap = Namespace(**lth_map),
                    keys = limits
                ))
                self.dmap = Namespace(**dtmp_map)
                self.save_current_run()
            else: logger.warning("No threhold peak detected!")
        else:
            limits = self.sol_properties.sol_chs_chbs.keys
            dtmp_map = {}
            data = self.sol_properties.sol_filter.filt_disk * self.sol_properties.sol_mask.n_mask
            for lim in limits:
                n_mask = np.zeros_like(data) * np.nan
                dtmp = np.copy(data)
                cv2.circle(n_mask, (int(self.resolution/2), int(self.resolution/2)), 
                           int(self.sol_properties.sol_radius.arcsec/self.rscale), 255, -1)
                n_mask[n_mask > 0] = 1
                dtmp = n_mask * dtmp
                dtmp[dtmp <= lim] = -1
                dtmp[dtmp > lim] = 0
                dtmp[dtmp == -1] = 1
                dtmp[np.isnan(dtmp)] = 0
                dtmp_map[str(lim)] = cv2.resize(np.copy(dtmp), (1024, 1024))
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
    
    def run_validation(self, rect=[(1200, 1200),(3000, 3000)]):
        """
        Validate against previous models
        """
        if not hasattr(self.sol_properties, "sol_validation"):
            logger.info("Run validation.")
            models = self.v_models.name
            keys = self.sol_properties.sol_chs_chbs.keys
            pmap = self.sol_properties.sol_chs_chbs.pmap
            output = {}
            for model in models:
                o = cutils.get_data_by_model(model)
                mask = np.zeros_like(o)
                cv2.rectangle(mask, rect[0], rect[1], 255, -1)
                sims = []
                for k in keys:
                    im = vars(self.dmap)[str(k)]
                    z = mask * cv2.resize(im, (self.resolution, self.resolution))
                    sims.append(cutils.measure_similaity(z, o)["cos"])
                smax, smax_args = np.max(sims), np.argmax(sims)
                k = keys[smax_args]
                im = cv2.resize(vars(self.dmap)[str(k)], (self.resolution, self.resolution))
                z = mask * im
                cutils.plot_compare_data(self.files.dir, z, o, model, k, "$\sigma\sim%.2f$"%smax, 
                                         vars(pmap)[str(k)])
                output[model] = Namespace(**dict(p=vars(pmap)[str(k)], lim=k))
                logger.warning(f"CHIPS validated against {model}")
            self.sol_properties.sol_validation = Namespace(**output)
            self.save_current_run()
        logger.info("Validation ended.")
        return
       
    def save_summary_plots(self):
        """
        Save all summary plot for analysis
        """
        summ = cutils.Summary(self, self.files.dir)
        summ.create_mask_filter_4by4_plot()
        summ.create_histogram_1by1_plot()
        summ.create_histogram_NbyM_plot()
        summ.create_histogram_NbyM_plot(slide=True)
        if hasattr(self, "dmap"): summ.create_CH_CHB_plots()
        return
    
    def save_current_run(self):
        """
        Save run to a .pickle file
        """
        fname = self.files.dir + "sol_properties.pickle"
        with open(fname, "wb") as h: pickle.dump(self.sol_properties, h, protocol=pickle.HIGHEST_PROTOCOL)
        return
    
    def load_last_run(self):
        """
        Load latest run to sol_properties
        """
        fname = self.files.dir + "sol_properties.pickle"
        if os.path.exists(fname):
            with open(fname, "rb") as h: self.sol_properties = pickle.load(h)
        else: self.save_current_run()
        return
    
    def clear_last_run(self):
        """
        Clear latest run
        """
        os.system(f"rm -rf {self.files.dir[:-1]}/*")
        return