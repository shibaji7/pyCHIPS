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

import datetime as dt
import glob
import os
from pathlib import Path
from typing import Any, List, Tuple

import aiapy.psf
import astropy
import astropy.units as u
import numpy as np
import requests
import sunpy.io
import sunpy.map
from aiapy.calibrate import (
    correct_degradation,
    normalize_exposure,
    register,
    update_pointing,
)
from loguru import logger
from sunpy.coordinates.sun import carrington_rotation_number, carrington_rotation_time
from sunpy.net import Fido, attrs


class SolarDisk(object):
    """An object subclass that holds all the information on solar disk.

    Attributes:
        date (datetime.datetime): Datetime of the solar disk '.fits' file.
        wavelength (int): Wave length of the disk image [171/193/211].
        resolution (int): Resolution of the image to work on [4096].
        apply_psf (bool): If `true`, conduct deconvololution with a point spread function.
        norm (bool): If `true`, conduct image registrtation and normalization.
        use_gpu (bool): If `true` and `cupy` is installed, do PSF deconvolution on the GPU with `cupy`.
        local_file (str): Regular expression that captures the local system file.
        desciption (str): Holds description of the '.fits' file.
        raw (sunpy.map.Map): Holds raw solar disk Map.
        psf (sunpy.map.Map): Holds deconvolved solar disk Map.
        normalized (sunpy.map.Map): Holds normalized solar disk Map (using raw or psf).
        pixel_radius (int): Holds solar disk radii in pixel.
    """

    def __init__(
        self,
        date: dt.datetime,
        wavelength: int,
        resolution: int = 4096,
        apply_psf: bool = False,
        norm: bool = True,
        use_gpu: bool = True,
        local_file: str = "sunpy/data/aia_lev1_{wavelength}a_{date_str}*.fits",
    ) -> None:
        """Initialize the parameters provided by kwargs."""
        self.date = date
        self.wavelength = wavelength
        self.resolution = resolution if resolution else 4096
        self.apply_psf = apply_psf
        self.norm = norm
        self.use_gpu = use_gpu
        self.local_file = local_file
        self.desciption = f"Solar Disk during {date} seeing through {wavelength}A, observing at {resolution} px."
        self.fetch()
        if self.norm:
            self.normalization()
        return

    def set_value(self, key: str, value: Any) -> None:
        """Methods to set an attribute inside `chips.fetch.SolarDisk` object.

        Arguments:
            key: Key/name of the attribute
            value: Value to set as attribute

        Returns:
            Method returns None
        """
        setattr(self, key, value)
        return

    def get_value(self, key: str) -> Any:
        """Methods to get an attribute from `chips.fetch.SolarDisk` object.

        Arguments:
            key: Key/name of the attribute

        Returns:
            Method returns a `object` if available
        """
        return getattr(self, key)

    def search_local(self) -> str:
        """Methods to search AIA disk '.fits' files in local system

        Arguments:

        Returns:
            Method returns a file name if file avilable.
        """
        logger.info("Searching local files ....")
        date_str = self.date.strftime("%Y_%m_%d")
        file_format = self.local_file.format(
            wavelength=self.wavelength, date_str=date_str
        )
        local_files = glob.glob(str(Path.home() / file_format))
        logger.info(f"Searching for local file format: {file_format}")
        local_file = local_files[0] if len(local_files) > 0 else None
        return local_file

    def fetch(self) -> None:
        """Methods to fetch AIA disk '.fits' files from remote 'JSOC' or local storage.

        Arguments:

        Returns:
            Method returns None
        """
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
            file_serach = Fido.fetch(q[0, 0])
            self.raw = sunpy.map.Map(file_serach)
        if self.resolution != 4096:
            pass
        if self.apply_psf:
            self.psf = aiapy.psf.deconvolve(self.raw, use_gpu=self.use_gpu)
        return

    def normalization(self) -> None:
        """Methods to search AIA files in local system

        Arguments:

        Returns:
            Method returns a file name if file avilable
        """
        key = "psf" if self.apply_psf else "raw"
        logger.info(
            f"Normalize map using L({self.wavelength}), R({self.resolution}) on {self.date}"
        )
        updated_point = update_pointing(getattr(self, key))
        registred = register(updated_point)
        registred = correct_degradation(registred)
        self.normalized = normalize_exposure(registred)
        self.normalized_data = np.copy(self.normalized.data)
        logger.info(f"Normalized value range: [{self.normalized_data.max()} <:> {self.normalized_data.min()}]")
        logger.info(f"Has normalized negative values: [{np.sum(self.normalized_data<0)==0}")
        if np.sum(self.normalized_data<0) > 0:
            self.normalized_data -= self.normalized_data.min() 
            self.normalized_data[self.normalized_data==0] = 0.01 # Issue with 0 values in log10
            logger.info(f"Rescaled value range: [{self.normalized_data.max()} <:> {self.normalized_data.min()}]")
        self.fetch_solar_parameters()
        return

    def fetch_solar_parameters(self) -> None:
        """Methods to fetch solar disk parameters. Sets solar radii in pixel.

        Arguments:

        Returns:
            Method returns None
        """
        logger.info("Extract solar basic properties.")
        self.rscale, self.r_sun, self.rsun_obs = (
            self.normalized._meta["cdelt2"],
            self.normalized._meta["r_sun"],
            self.normalized._meta["rsun_obs"],
        )
        self.pixel_radius = int(self.rsun_obs / self.rscale)

        self.log_normalized_data = np.log10(self.normalized_data)
        logger.info(f"Log value range: [{self.log_normalized_data.max()} <:> {self.log_normalized_data.min()}]")
        return

    def plot_disk_images(
        self,
        types: List[str] = ["raw", "normalized"],
        figsize: Tuple[int] = (6, 3),
        dpi: int = 240,
        nrows: int = 1,
        ncols: int = 2,
        fname: str = None,
    ) -> None:
        """Plotting method to generate diagonestics plots

        Arguments:
            types: Name of the map attributes to plot ['raw','normalized','psf'].
            figsize: Size of the figure.
            dpi: Figure DPI.
            nrows: Number of rows in the figure.
            ncols: Number of columns in the figure.
            fname: File name of the figure (expected file format: png, bmp, pdf, jpg).

        Returns:
            Method returns None
        """
        plotting_types = [t for t in types if hasattr(self, t)]
        from plots import Annotation, ImagePalette

        im = ImagePalette(
            figsize=figsize,
            dpi=dpi,
            nrows=nrows,
            ncols=ncols,
        )
        for t in plotting_types:
            im.draw_colored_disk(
                self.get_value(t), self.pixel_radius, resolution=self.resolution
            )
        annotations = []
        annotations.append(
            Annotation(
                self.disk.date.strftime("%Y-%m-%d %H:%M"), 0.05, 1.05, "left", "center"
            )
        )
        annotations.append(
            Annotation(
                r"$\lambda=%d\AA$" % self.disk.wavelength,
                -0.05,
                0.99,
                "center",
                "top",
                rotation=90,
            )
        )
        im.annotate(annotations)
        if fname:
            im.save(fname)
        im.close()
        return


class RegisterAIA(object):
    """An object superclass that holds all the information on `chips.fetch.SolarDisk`.

    Attributes:
        date (datetime.datetime): Datetime of the solar disk '.fits' file.
        wavelengths (List[int]): List of wavelength of the disk image [171/193/211] to be run.
        resolution (int): Resolutions of the image to work on 4096 (4k).
        apply_psf (bool): If `true`, conduct deconvololution with a point spread function.
        norm (bool): If `true`, conduct image registrtation and normalization.
        local_file (str): Regular expression that captures the local system file.
        save_dir (str): Save directory local for dataset (filtered and analyzed).
    """

    def __init__(
        self,
        date: dt.datetime,
        wavelengths: List[int] = [171, 193, 211],
        resolution: int = 4096,
        apply_psf: bool = False,
        norm: bool = True,
        local_file: str = "sunpy/data/aia_lev1_{wavelength}a_{date_str}*.fits",
        save_dir: str = "tmp/",
    ) -> None:
        self.wavelengths = wavelengths
        self.date = date
        self.resolution = resolution
        self.apply_psf = apply_psf
        self.norm = norm
        self.local_file = local_file
        self.save_dir = save_dir + f"chips_output/{date.strftime('%Y-%m-%d-%H-%M')}/"
        os.makedirs(self.save_dir, exist_ok=True)
        self.datasets = dict()
        for wv in wavelengths:
            self.datasets[wv] = dict()
            self.datasets[wv][self.resolution] = SolarDisk(
                self.date,
                wv,
                self.resolution,
                apply_psf=self.apply_psf,
                norm=self.norm,
                local_file=self.local_file,
            )
        return
    
    def plot_scatter_maps(
            self,
            figsize: Tuple[int] = (6, 9),
            dpi: int = 240,
            nrows: int = 3,
            ncols: int = 2,
            fname: str = None,
            scale: str = "linear",
        ) -> None:
        """Plotting method to generate scatter

        Arguments:
            figsize: Size of the figure.
            dpi: Figure DPI.
            nrows: Number of rows in the figure.
            ncols: Number of columns in the figure.
            fname: File name of the figure (expected file format: png, bmp, pdf, jpg).

        Returns:
            Method returns None
        """
        from plots import Annotation, ImagePalette

        ip = ImagePalette(
            figsize=figsize,
            dpi=dpi,
            nrows=nrows,
            ncols=ncols,
        )
        
        disk193, disk171, disk211 = (
            self.datasets[193][self.resolution],
            self.datasets[171][self.resolution],
            self.datasets[211][self.resolution]
        )
        x, y, z = (
            disk193.log_normalized_data.ravel(),
            disk171.log_normalized_data.ravel(),
            disk211.log_normalized_data.ravel(),
        ) if scale=="log" else \
                (
                    disk193.normalized_data.ravel(),
                    disk171.normalized_data.ravel(),
                    disk211.normalized_data.ravel(), 
                )
        if scale=="log":
            import pandas as pd
            data = pd.DataFrame()
            data["x"], data["y"], data["z"] = (x, y, z)
            data = data.dropna()
            x, y, z = np.array(data.x), np.array(data.y), np.array(data.z)

        import matplotlib
        ax = ip.__axis__(axis_off=False)
        ax.scatter(x, y, color="k", s=0.01)
        ax.set_xlim([0, 2] if scale=="log" else [0, 300])
        ax.set_ylim([0, 2] if scale=="log" else [0, 300])
        ax.set_ylabel(r"$\log_{10}(I_{171})$" if scale=="log" else r"$I_{171}$")
        ax.set_xlabel(r"$\log_{10}(I_{193})$" if scale=="log" else r"$I_{193}$")

        ax = ip.__axis__(axis_off=False)
        H, xedge, yedge, _ = ax.hist2d(x, y, bins=1000, norm=matplotlib.colors.LogNorm(), density=True)
        valleys = find_valleys(H)
        #print(valleys)
        ax.set_xlim([0, 2] if scale=="log" else [0, 300])
        ax.set_ylim([0, 2] if scale=="log" else [0, 300])
        ax.set_xlabel(r"$\log_{10}(I_{193})$" if scale=="log" else r"$I_{193}$")
        ax.plot(valleys[0], valleys[1], ls="-", lw=0.8, color="r")

        ax = ip.__axis__(axis_off=False)
        ax.scatter(x, z, color="k", s=0.01)
        ax.set_xlim([0, 2] if scale=="log" else [0, 300])
        ax.set_ylim([0, 2] if scale=="log" else [0, 300])
        ax.set_xlabel(r"$\log_{10}(I_{193})$" if scale=="log" else r"$I_{193}$")
        ax.set_ylabel(r"$\log_{10}(I_{211})$" if scale=="log" else r"$I_{211}$")

        ax = ip.__axis__(axis_off=False)
        ax.hist2d(x, z, bins=1000, norm=matplotlib.colors.LogNorm(), density=True)
        ax.set_xlim([0, 2] if scale=="log" else [0, 300])
        ax.set_ylim([0, 2] if scale=="log" else [0, 300])
        ax.set_xlabel(r"$\log_{10}(I_{193})$" if scale=="log" else r"$I_{193}$")

        ax = ip.__axis__(axis_off=False)
        ax.scatter(z, y, color="k", s=0.01)
        ax.set_xlim([0, 2] if scale=="log" else [0, 300])
        ax.set_ylim([0, 2] if scale=="log" else [0, 300])
        ax.set_xlabel(r"$\log_{10}(I_{211})$" if scale=="log" else r"$I_{211}$")
        ax.set_ylabel(r"$\log_{10}(I_{171})$" if scale=="log" else r"$I_{171}$")

        ax = ip.__axis__(axis_off=False)
        ax.hist2d(z, y, bins=1000, norm=matplotlib.colors.LogNorm(), density=True)
        ax.set_xlim([0, 2] if scale=="log" else [0, 300])
        ax.set_ylim([0, 2] if scale=="log" else [0, 300])
        ax.set_xlabel(r"$\log_{10}(I_{211})$" if scale=="log" else r"$I_{211}$")

        annotations = []
        annotations.append(
            Annotation(
                self.date.strftime("%Y-%m-%d %H:%M"), 0.05, 1.05, "left", "center"
            )
        )
        ip.annotate(annotations)
        fname = fname if fname else self.save_dir + "segmentation_scatter.png"
        ip.save(fname, hspace=0.1, wspace=0.1)
        ip.close()
        return


class SynopticMap(object):
    """An object class that holds all the information on solar synoptic maps.

    Attributes:
        date (datetime.datetime): Datetime of the solar disk '.fits' file.
        CR_equivalance (int): Carrington rotation number (if not provided then computed by date)
        wavelength (int): Wave length of the disk image [171/193/211].
        desciption (str): Holds description of the '.fits' file.
        location (str): Local file store to save the synoptic maps.
        base_url_regex (str): Regex of the JSOC/SDO repository url.
        file_name_regex (str): Regex of the local file name.
        raw (sunpy.map.Map): Holds raw solar disk Map.
    """

    def __init__(
        self,
        date: dt.datetime,
        CR_equivalance: int = None,
        wavelength: int = 193,
        location: str = "sunpy/data/synoptic/",
        base_url_regex: str = "https://sdo.gsfc.nasa.gov/assets/img/synoptic/AIA{:04d}/CR{:04d}.fits",
        file_name_regex: str = "AIA0{:04d}_CR{:04d}.fits",
    ) -> None:
        """Initalization method"""
        self.date = date
        self.wavelength = wavelength
        self.location = location
        self.base_url_regex = base_url_regex
        if CR_equivalance:
            self.CR_equivalance = CR_equivalance
            self.date = carrington_rotation_time(self.CR_equivalance)
        else:
            self.CR_equivalance = np.ceil(carrington_rotation_number(self.date)).astype(
                int
            )
        logger.info(f"Carrington Rotation Number: {self.CR_equivalance}")
        # Create all the urls and file names
        self.fname = file_name_regex.format(wavelength, self.CR_equivalance)
        self.remote_file_url = base_url_regex.format(wavelength, self.CR_equivalance)
        logger.info(f"Remote file {self.remote_file_url}")
        self.local_file_dir = str(Path.home() / location)
        os.makedirs(self.local_file_dir, exist_ok=True)
        self.local_file = self.local_file_dir + "/" + self.fname
        logger.info(f"Local file {self.local_file}")
        self.fetch()
        return

    def fetch(self) -> None:
        """Method to download and store the remote files to local folder and fetch it into `sunpy.map.Map` object.

        Attributes:

        Returns:
            Method returns None
        """
        if not os.path.exists(self.local_file):
            req = requests.get(self.remote_file_url)
            logger.info(f"Fetching remore file status code:{req.status_code}")
            if req.status_code == 200:
                with open(self.local_file, "wb") as f:
                    f.write(req.content)
        with astropy.io.fits.open(self.local_file) as hdul:
            hdul[0].verify("fix")
            hdul[0].header["cunit1"] = "arcsec"
            hdul[0].header["cunit2"] = "arcsec"
            self.raw = sunpy.map.Map(hdul[0].data, hdul[0].header)
        return

    def set_value(self, key: str, value: Any) -> None:
        """Methods to set an attribute inside `chips.fetch.SynopticMap` object.

        Arguments:
            key: Key/name of the attribute
            value: Value to set as attribute

        Returns:
            Method returns None
        """
        setattr(self, key, value)
        return

    def get_value(self, key: str) -> Any:
        """Methods to get an attribute from `chips.fetch.SynopticMap` object.

        Arguments:
            key: Key/name of the attribute

        Returns:
            Method returns a `object` if available
        """
        return getattr(self, key)

def find_valleys(arr, size=11):
    from scipy.ndimage import minimum_filter
    # Apply minimum filter to find local minima
    min_filtered = minimum_filter(arr, size=size)
    # Compare original array with filtered array to find local minima
    valleys = np.where(arr == min_filtered)
    return list(zip(valleys[0], valleys[1]))