"""
    fetch.py: Module is used to fetch the data from standford system.
"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"


import aiapy.psf
import astropy.units as u
import sunpy.io
import sunpy.map
from aiapy.calibrate import normalize_exposure, register, update_pointing
from loguru import logger
from sunpy.net import Fido, attrs


class RegisterAIA(object):
    """
    Regsitered all AIA dataset
    """

    def __init__(
        self, date, wavelengths=[193], resolution=4096, vmins=[10], apply_psf=False
    ):
        self.wavelengths = wavelengths
        self.date = date
        self.resolution = resolution
        self.vmins = vmins
        self.apply_psf = apply_psf
        self.data = dict()
        self.fetch()
        self.normalized()
        return

    def fetch(self):
        for wv in self.wavelengths:
            self.data[wv] = dict()
            logger.info(f"Fetching({wv}), R({self.resolution}) on {self.date}")
            q = Fido.search(
                attrs.Time(
                    self.date.strftime("%Y-%m-%dT%H:%M:%S"),
                    (self.date + dt.timedelta(seconds=60)).strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    ),
                ),
                attrs.Instrument("AIA"),
                attrs.Wavelength(
                    wavemin=wv * u.angstrom,
                    wavemax=wv * u.angstrom,
                ),
            )
            logger.info(f"Record found: len({len(q)})")
            logger.info(f"Registering FITS data")
            self.data[wv][self.resolution] = dict(
                raw=sunpy.map.Map(Fido.fetch(q[0, 0]))
            )
            if self.apply_psf:
                self.data[wv][self.resolution]["psf"] = aiapy.psf.deconvolve(
                    self.data[wv][self.resolution]["raw"]
                )
        return

    def normalized(self):
        key = "psf" if self.apply_psf else "raw"
        for wv in self.wavelengths:
            logger.info(
                f"Normalize map using L({wv}), R({self.resolution}) on {self.date}"
            )
            updated_point = update_pointing(self.data[wv][self.resolution][key])
            registred = register(updated_point)
            self.data[wv][self.resolution]["normalized"] = normalize_exposure(registred)
        return
