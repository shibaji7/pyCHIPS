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
from matplotlib.collections import PatchCollection

import os
import datetime as dt

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
    
    def __init__(self, _dict_, local="tmp/{date}/"):
        self._dict_ = _dict_
        self.aia = RegisterAIA(_dict_["date"], _dict_["wavelength"], _dict_["resolution"], _dict_["vmin"])
        self.folder = local.format(date=_dict_["date"].strftime("%Y-%m-%d-%H-%M"))
        if not os.path.exists(self.folder): os.system("mkdir -p " + self.folder)
        self.saveimg()
        return
    
    def saveimg(self):
        patches = []
        file = self.folder + "01_raw.png"
        fig = plt.figure()
        ax = fig.add_subplot(211)
        self.aia.m_normalized.plot(annotate=False, axes=ax, vmin=self._dict_["vmin"])
        self.aia.m_normalized.draw_limb()
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(212)
        self.aia.m_normalized.plot(annotate=False, axes=ax, vmin=self._dict_["vmin"])
        #patches.append(Circle((2084, 2084), r))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(file, bbox_inches="tight")
        return
    
    def estimate_thresholds_on_solar_disk(self):
        
        return
    
if __name__ == "__main__":
    _dict_ = {}
    _dict_["date"] = dt.datetime(2018,5,30,12)
    _dict_["wavelength"] = 193
    _dict_["resolution"] = 4096
    _dict_["vmin"] = 35
    CHIPS(_dict_)