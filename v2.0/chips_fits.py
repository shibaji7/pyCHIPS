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
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm

import os
import datetime as dt
import numpy as np
from scipy import signal
import cv2

import astropy.units as u
from sunpy.net import Fido, attrs
import sunpy.map
from aiapy.calibrate import register, update_pointing, normalize_exposure

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
        self._dict_ = _dict_
        self.aia = RegisterAIA(_dict_["date"], _dict_["wavelength"], _dict_["resolution"], _dict_["vmin"])
        self.folder = local.format(date=_dict_["date"].strftime("%Y-%m-%d-%H-%M"))
        if not os.path.exists(self.folder): os.system("mkdir -p " + self.folder)
        self.rscale = rscale
        self.stage01analysis()
        return
    
    def stage01analysis(self, plot=True):
        self.properties = {}
        rsun = self.aia.m_normalized.rsun_obs.value
        mask = np.zeros_like(self.aia.m_normalized.data)
        mask[:] = np.nan
        cv2.circle(mask, (2048,2048), int(rsun/self.rscale), 255, -1)
        mask[np.isnan(mask)] = 0.
        mask[mask > 1] = 1.
        disk_mask = np.copy(mask)
        disk_data = mask * self.aia.m_normalized.data
        disk_filt_data = mask * signal.medfilt2d(self.aia.m_normalized.data, self._dict_["medfilt.kernel"])
        self.["radius"] = {"arcsec": rsun, "pix": int(rsun/self.rscale)}
        self.["disk"] = {"mask": np.copy(mask), 
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
            ax.imshow(mask, origin="lower", cmap="gray")
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
    
    def stage02analysis(self, plot=True):
        
        
        return
    
if __name__ == "__main__":
    _dict_ = {}
    _dict_["date"] = dt.datetime(2018,5,30,12)
    _dict_["wavelength"] = 193
    _dict_["resolution"] = 4096
    _dict_["vmin"] = 35
    _dict_["medfilt.kernel"] = 3
    CHIPS(_dict_)