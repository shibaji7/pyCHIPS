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

import astropy.units as u
from sunpy.net import Fido, attrs
import sunpy.map
from aiapy.calibrate import register, update_pointing, normalize_exposure

import chips_utils as cutils

class RegisterAIA(object):
    
    def __init__(self, date, wavelength=193, resolution=4096, vmin=10):
        self.wavelength = wavelength
        self.date = date
        self.resolution = resolution
        self.vmin = vmin
        self.normalized()
        return
    
    def normalized(self):
        q = Fido.search(
            attrs.Time(self.date.strftime("%Y-%m-%dT%H:%M:%S"), (self.date + dt.timedelta(seconds=11)).strftime("%Y-%m-%dT%H:%M:%S")),
            attrs.Instrument("AIA"),
            attrs.Wavelength(wavemin=self.wavelength*u.angstrom, wavemax=self.wavelength*u.angstrom),
        )
        self.m = sunpy.map.Map(Fido.fetch(q[0,0]))
        m_updated_pointing = update_pointing(self.m)
        m_registered = register(m_updated_pointing)
        self.m_normalized = normalize_exposure(m_registered)
        return
    
class CHIPS(object):
    
    def __init__(self, _dict_, local="tmp/{date}/", rscale=0.6):
        """
        Get the AIA data
        """
        self._dict_ = _dict_
        for k in self._dict_.keys():
            setattr(self, k, self._dict_[k])
        self.aia = RegisterAIA(_dict_["date"], _dict_["wavelength"], _dict_["resolution"], _dict_["vmin"])
        self.folder = local.format(date=_dict_["date"].strftime("%Y-%m-%d-%H-%M"))
        if not os.path.exists(self.folder): os.system("mkdir -p " + self.folder)
        self.rscale = rscale
        self.stage01analysis()
        self.stage02analysis()
        self.stage03analysis()
        self.stage04Validation()
        return
    
    def stage01analysis(self, plot=False):
        """
        Convert raw fit level data into 2D grayscale image level data
        """
        self.properties = {}
        rsun = self.aia.m_normalized.rsun_obs.value
        mask = np.zeros_like(self.aia.m_normalized.data)
        mask[:] = np.nan
        cv2.circle(mask, (int(self.resolution/2), int(self.resolution/2)), int(rsun/self.rscale), 255, -1)
        mask[mask > 1] = 1.
        dmask = np.copy(mask)
        dmask[np.isnan(dmask)] = 0.
        dmask[dmask > 1] = 1.
        disk_mask = np.copy(dmask)
        disk_data = dmask * self.aia.m_normalized.data
        disk_filt_data = dmask * signal.medfilt2d(self.aia.m_normalized.data, self._dict_["medfilt.kernel"])
        self.properties["radius"] = {"arcsec": rsun, "pix": int(rsun/self.rscale)}
        self.properties["disk"] = {"mask": np.copy(mask), "dmask": np.copy(dmask), 
                         "raw": {"data": self.aia.m_normalized.data, "disk": disk_data},
                         "filter": {"data": signal.medfilt2d(self.aia.m_normalized.data, self._dict_["medfilt.kernel"]), 
                                    "disk": disk_filt_data},
                        }
        
        if plot:
            file = self.folder + "01_analysis.png"
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(221)
            self.aia.m_normalized.plot(annotate=False, axes=ax, vmin=self._dict_["vmin"])
            self.aia.m_normalized.draw_limb()
            ax.set_xticks([])
            ax.set_yticks([])
            ax = fig.add_subplot(222)
            ax.imshow(dmask, origin="lower", cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            ax = fig.add_subplot(223)
            ux = np.copy(disk_data)
            ux[ux <= 0.] = 1
            ux = np.log10(ux)
            ax.imshow(ux, origin="lower", cmap="gray", vmin=1, vmax=3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax = fig.add_subplot(224)
            ux = np.copy(disk_filt_data)
            ux[ux <= 0.] = 1
            ux = np.log10(ux)
            ax.imshow(ux, origin="lower", cmap="gray", vmin=1, vmax=3)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.subplots_adjust(wspace=0.1, hspace=0.1)
            fig.savefig(file, bbox_inches="tight")
        return
    
    def stage02analysis(self, plot=False, n=5000, ht=None):
        """
        Find the 1st peaks of the histograms (stage 2 processes)
        """
        data = self.properties["disk"]["filter"]["disk"] * self.properties["disk"]["mask"]
        d = data.ravel()[~np.isnan(data.ravel())] 
        h, be = np.histogram(d, bins=n, density=True)
        if ht is None: ht = np.max(h)/5
        peaks, _ = signal.find_peaks(h, height=ht)
        bc = be[:-1] + np.diff(be)
        self.peak_points = bc[peaks]
        if plot:
            file = self.folder + "02_analysis.png"
            fig = plt.figure(figsize=(3,3), dpi=150)
            ax = fig.add_subplot(111)
            h, be, _ = ax.hist(data.ravel(), bins=n, histtype="step", color="r", density=True)
            ax.axvline(self.peak_points[0], color="b", ls="--", lw=0.8)
            ax.set_xscale("log")
            ax.set_xlabel(r"Intensity, $I_s$")
            ax.set_ylabel(r"Density, $D(I_s)$")
            fig.savefig(file, bbox_inches="tight")
        return
    
    def calculate_prob(self, dm, lth, l0):
        dm = dm[~np.isnan(dm)]
        dm = dm[dm <= lth]
        ps = 1 / ( 1 + np.exp(dm-l0).round(4) )
        h, e = np.histogram(ps,bins=np.linspace(0,1,21))
        b = (e + np.diff(e)[0]/2)[:-1]
        p = np.sum(b[b>.5]*h[b>.5])/np.sum(b*h)
        return np.round(p,4)
    
    def calculate_entropy(self, dm, lth, l0):
        dm = dm[~np.isnan(dm)]
        dm_low, dm_high = dm[dm <= lth], dm[dm > lth]
        ps_low, _ = np.histogram(dm_low, bins=21, density=True)
        ps_high, _ = np.histogram(dm_high, bins=21, density=True)
        ps_low, ps_high = np.round(ps_low, 3), np.round(ps_high, 3)
        #1 / ( 1 + np.exp(dm_low-l0).round(4) ), 1 / ( 1 + np.exp(l0-dm_high).round(4) )
        e_ratio = np.round(stats.entropy(ps_low), 3), np.round(stats.entropy(ps_high), 3)
        return e_ratio
    
    def stage03analysis(self, plot=False, n=5000, ht=1e-5):
        """
        Find the CHs and CHBs and associated probability
        """
        l0 = self.peak_points[0] 
        limits = np.arange(5,50)
        data = self.properties["disk"]["filter"]["disk"] * self.properties["disk"]["mask"]
        self.tmp_map = {}
        self.prob_map = {}
        self.entropy_map = {}
        self.lth_map = {}
        for lim in limits:
            lth = l0 + lim
            self.lth_map[lim] = lth
            tmp = np.copy(data)
            mask = np.zeros_like(data)
            mask[:] = np.nan
            cv2.circle(mask, (int(self.resolution/2), int(self.resolution/2)), 
                       int(self.properties["radius"]["arcsec"]/self.rscale), 255, -1)
            mask[mask > 0] = 1
            tmp = mask * tmp
            tmp[tmp<=lim] = -1
            tmp[tmp>lim] = 0
            tmp[tmp==-1] = 1
            tmp[np.isnan(tmp)] = 0
            self.tmp_map[lim] = np.copy(tmp)
            dm = np.copy(data).ravel()
            p = self.calculate_prob(dm, lth, l0)
            e = self.calculate_entropy(dm, lth, l0)
            self.prob_map[lim] = r"$\theta=%.3f$"%p+"\n"+r"$I_{th}=%d$"%lim
            print(" Limit:", lim, p, e)
        
        if plot:
            file = self.folder + "03_analysis.png"
            k = 0
            fig, axes = plt.subplots(dpi=120, figsize=(9,9), nrows=4, ncols=4, sharex="all", sharey="all")
            keys = list(self.tmp_map.keys())
            cmap = matplotlib.cm.gray
            cmap.set_bad("k",1.)
            for i in range(4):
                for j in range(4):
                    ax = axes[i,j]
                    ax.imshow(self.tmp_map[keys[k]], origin="lower", cmap=cmap)
                    ax.set_yticks([])
                    ax.set_xticks([])
                    k += 2
                    ax.text(0.8, 0.8, self.prob_map[keys[k]], ha="center", va="center", transform=ax.transAxes,
                            fontdict={"size":7, "color":"w"})
                    print(" Limits - ", keys[k])
            fig.subplots_adjust(wspace=0.01, hspace=0.05)
            fig.savefig(file, bbox_inches="tight")
        return
    
    def reshape(self, x, scale_percent=50.):
        width = int(x.shape[1] * scale_percent / 100)
        height = int(x.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        rs = cv2.resize(x, dim)
        return rs
    
    def stage04Validation(self, rect=[(1200, 1200),(3000, 3000)], plot=True):
        """
        Validation 
        """
        models = ["ASSA", "CHARM", "CHIMERA", "CHORTLE", "CNN", "HAMADA", "JAROLIM",
                  "SPOCA", "TH35"]
        keys = list(self.tmp_map.keys())
        for m in models:
            o = cutils.get_data_by_model(m)
            mask = np.zeros_like(o)
            cv2.rectangle(mask, rect[0], rect[1], 255, -1)
            sims = []
            for k in keys:
                im = self.tmp_map[k]
                z = mask * im
                sims.append(cutils.measure_similaity(z, o)["cos"])
            smax, smax_args = np.max(sims), np.argmax(sims)
            if plot: 
                k = keys[smax_args]
                im = self.tmp_map[k]
                z = mask * im
                cutils.plot_compare_data(self.folder, z, o, m, k, "$\sigma\sim%.2f$"%smax, self.prob_map[k])
        return
    
if __name__ == "__main__":
    _dict_ = {}
    _dict_["date"] = dt.datetime(2018,5,30,12)
    _dict_["wavelength"] = 193
    _dict_["resolution"] = 4096
    _dict_["vmin"] = 35
    _dict_["medfilt.kernel"] = 3
    CHIPS(_dict_)