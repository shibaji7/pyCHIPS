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
import datetime as dt


class SolarDisk(object):
    """A simple object subclass that holds all the informations on solar disk and.
    
    Methods:
        normalized
        fetch
        set_value
        get_value
    """
    
    def __init__(
        self, 
        date: dt.datetime, 
        wavelength: int, 
        resolution: int=4096,
        apply_psf: bool=False,
        norm: bool=True,
    ) -> None:
        self.date = date
        self.wavelength = wavelength
        self.resolution = resolution if resolution else 4096
        self.apply_psf = apply_psf
        self.norm = norm
        self.desciption = f"Solar Disk during {date} seeing through {wavelength}A, observing at {resolution} px."
        self.fetch()
        if self.norm: self.normalized()
        return
    
    def set_value(self, key: str, value: object) -> None:
        setattr(self, key, value)
        return
    
    def get_value(self, key: str) -> object:
        return getattr(self, key)

    def fetch(self):
        logger.info(f"Fetching({self.wavelength}), R({self.resolution}) on {self.date}")
        q = Fido.search(
            attrs.Time(
                self.date.strftime("%Y-%m-%dT%H:%M:%S"),
                (self.date + dt.timedelta(seconds=60)).strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
            ),
            attrs.Instrument("AIA"),
            attrs.Wavelength(
                wavemin=self.wavelength * u.angstrom,
                wavemax=self.wavelength * u.angstrom,
            ),
        )
        logger.info(f"Record found: len({len(q)})")
        logger.info(f"Registering FITS data")
        self.raw = sunpy.map.Map(Fido.fetch(q[0, 0]))
        if self.apply_psf: self.psf = aiapy.psf.deconvolve(self.raw)
        return

    def normalized(self):
        key = "psf" if self.apply_psf else "raw"
        logger.info(f"Normalize map using L({self.wavelength}), R({self.resolution}) on {self.date}")
        updated_point = update_pointing(
            getattr(self, key)
        )
        self.registred = register(updated_point)
        self.normalized = normalize_exposure(self.registred)
        return


class RegisterAIA(object):
    """
    Regsitered all AIA dataset
    """

    def __init__(
        self, 
        date,
        wavelengths=[193],
        resolutions=[4096],
        apply_psf=False,
        fetch=True, 
        norm=True
    ):
        self.wavelengths = wavelengths
        self.date = date
        self.resolutions = resolutions
        self.apply_psf = apply_psf
        self.datasets = dict()
        for wv in wavelengths:
            self.datasets[wv] = dict()
            for res in resolutions:
                self.datasets[wv][res] = SolarDisk(
                    self.date, wv, res,
                    apply_psf, norm
                )
        # self.data = dict()
        # if fetch:
        #     self.fetch()
        # if norm:
        #     self.normalized()
        return

    # def fetch(self):
    #     for wv in self.wavelengths:
    #         self.data[wv] = dict()
    #         logger.info(f"Fetching({wv}), R({self.resolution}) on {self.date}")
    #         q = Fido.search(
    #             attrs.Time(
    #                 self.date.strftime("%Y-%m-%dT%H:%M:%S"),
    #                 (self.date + dt.timedelta(seconds=60)).strftime(
    #                     "%Y-%m-%dT%H:%M:%S"
    #                 ),
    #             ),
    #             attrs.Instrument("AIA"),
    #             attrs.Wavelength(
    #                 wavemin=wv * u.angstrom,
    #                 wavemax=wv * u.angstrom,
    #             ),
    #         )
    #         logger.info(f"Record found: len({len(q)})")
    #         logger.info(f"Registering FITS data")
    #         self.data[wv][self.resolution] = dict(
    #             raw=sunpy.map.Map(Fido.fetch(q[0, 0]))
    #         )
    #         if self.apply_psf:
    #             self.data[wv][self.resolution]["psf"] = aiapy.psf.deconvolve(
    #                 self.data[wv][self.resolution]["raw"]
    #             )
    #     return

    # def normalized(self):
    #     key = "psf" if self.apply_psf else "raw"
    #     for wv in self.wavelengths:
    #         logger.info(
    #             f"Normalize map using L({wv}), R({self.resolution}) on {self.date}"
    #         )
    #         updated_point = update_pointing(self.data[wv][self.resolution][key])
    #         registred = register(updated_point)
    #         self.data[wv][self.resolution]["normalized"] = normalize_exposure(registred)
    #     return
