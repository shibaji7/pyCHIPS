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

from typing import List, Any
import aiapy.psf
import astropy.units as u
import sunpy.io
import sunpy.map
from aiapy.calibrate import normalize_exposure, register, update_pointing
from loguru import logger
from sunpy.net import Fido, attrs
import datetime as dt
import glob
from pathlib import Path


class SolarDisk(object):
    """A simple object subclass that holds all the informations on solar disk and.
    
    Methods:
        normalization
        fetch
        set_value
        get_value
        normalization
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
        if self.norm: self.normalization()
        return
    
    def set_value(self, key: str, value: object) -> None:
        setattr(self, key, value)
        return
    
    def get_value(self, key: str) -> Any:
        return getattr(self, key)

    def search_local(self) -> str:
        logger.info("Searching local files ....")
        date_str = self.date.strftime("%Y_%m_%d")
        local_files = glob.glob(
            str(
                Path.home() / f"sunpy/data/aia_lev1_{self.wavelength}a_{date_str}*.fits"
            )
        )
        local_file = local_files[0] if len(local_files) > 0 else None
        return local_file

    def fetch(self) -> None:
        logger.info(f"Fetching({self.wavelength}), R({self.resolution}) on {self.date}")
        local_file = self.search_local()
        if local_file:
            logger.info(f"Load local file {local_file}")
            self.raw = sunpy.map.Map(local_file)
        else:
            logger.info("No local files...")
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
        if self.resolution!=4096: pass
        if self.apply_psf: self.psf = aiapy.psf.deconvolve(self.raw)
        return

    def normalization(self) -> None:
        key = "psf" if self.apply_psf else "raw"
        logger.info(f"Normalize map using L({self.wavelength}), R({self.resolution}) on {self.date}")
        updated_point = update_pointing(
            getattr(self, key)
        )
        self.registred = register(updated_point)
        self.normalized = normalize_exposure(self.registred)
        self.fetch_solar_parameters()
        return

    def fetch_solar_parameters(self) -> None:
        logger.info("Extract solar basic properties.")
        self.rscale, self.r_sun, self.rsun_obs = (
            self.normalized._meta["cdelt2"],
            self.normalized._meta["r_sun"],
            self.normalized._meta["rsun_obs"]
        )
        self.pixel_radius = int(self.rsun_obs/self.rscale)
        return


class RegisterAIA(object):
    """
    Regsitered all AIA dataset
    """

    def __init__(
        self, 
        date: dt.datetime,
        wavelengths: List[int]=[193],
        resolutions: List[int]=[4096],
        apply_psf: bool=False,
        norm: bool=True,
    ) -> None:
        self.wavelengths = wavelengths
        self.date = date
        self.resolutions = resolutions
        self.apply_psf = apply_psf
        self.norm = norm
        self.datasets = dict()
        for wv in wavelengths:
            self.datasets[wv] = dict()
            for res in resolutions:
                self.datasets[wv][res] = SolarDisk(
                    self.date, wv, res,
                    apply_psf=self.apply_psf,
                    norm=self.norm
                )
        return