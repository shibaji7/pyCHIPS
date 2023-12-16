"""
    chips.py: Module is used to implement edge detection tecqniues using thresholding.
"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
from argparse import Namespace
from typing import List

import cv2
import fetch
import numpy as np
from loguru import logger
from plots import ChipsPlotter
from scipy import signal


class Chips(object):
    r"""An object class that runs the CHIPS algorithm step-by-step with a set of input parameters.

    Attributes:
        aia (chips.fetch.RegisterAIA): AIA method that holds all the information on `List[fetch.SolarDisk]`.
        base_folder (str): Base folder to store the processed figures and datasets.
        medfilt_kernel (int): Median filtering Gaussian kernel size (odd number).
        h_bins (int): Number of bins in histogram while running Otsu's method.
        h_thresh (float): Otsu's thresold (computed inside if not given, $I_{th}$).
        ht_peak_ratio (int): Needed to cpmpute threhold while computing Otsu's method.
        hist_xsplit (int): Needed to compute thresholds by splitted image (number of splitted along the image width).
        hist_ysplit (int): Needed to compute thresholds by splitted image (number of splitted along the image height).
        threshold_range (List[float]): List of thresholds from `h_thresh` to run CHIPS.
        porb_threshold (int): This is $x_{\tau}$.
    """

    def __init__(
        self,
        aia,
        base_folder: str = "tmp/chips_dataset/",
        medfilt_kernel: int = 3,
        h_bins: int = 5000,
        h_thresh: float = None,
        ht_peak_ratio: int = 5,
        hist_xsplit: int = 2,
        hist_ysplit: int = 2,
        threshold_range: List[float] = [0, 20],
        porb_threshold: float = 0.8,
    ) -> None:
        """Initialization method
        """
        self.aia = aia
        self.base_folder = base_folder
        self.medfilt_kernel = medfilt_kernel
        self.h_bins = h_bins
        self.h_thresh = h_thresh
        self.ht_peak_ratio = ht_peak_ratio
        self.hist_xsplit = hist_xsplit
        self.hist_ysplit = hist_ysplit
        self.threshold_range = threshold_range
        self.porb_threshold = porb_threshold
        self.get_base_folder()
        return

    def get_base_folder(self) -> None:
        """Method to find local base folder to store figures and outputs.
        
        Attributes:
        
        Returns:
            Method returns None.
        """
        self.folder = self.base_folder + self.aia.date.strftime("%Y.%m.%d")
        os.makedirs(self.folder, exist_ok=True)
        return

    def run_CHIPS(
        self,
        wavelength: int=None,
        resolution: int=None,
    ) -> None:
        """This method selects the `chips.fetch.SolarDisk`(s) objects and runs the CHIPS algorithm sequentially.

        Attributes:
            wavelength (int): Wave length of the disk image [171/193/211].
            resolution (int): Resolution of the image to work on [4096].

        Returns:
            Method returns None.
        """
        self.simulaion_outputs = dict()
        if (wavelength is not None) and (resolution is not None):
            if (wavelength in self.aia.wavelengths) and (
                resolution in self.aia.resolutions
            ):
                self.simulaion_outputs[wavelength] = dict()
                logger.info(f"Singleton: Running CHIPS for {wavelength}/{resolution}")
                disk = self.aia.datasets[wavelength][resolution]
                self.simulaion_outputs[wavelength][resolution] = self.process_CHIPS(
                    disk
                )
            else:
                logger.error(f"No entry for {wavelength} and {resolution}!!!")
        else:
            for wavelength in self.aia.wavelengths:
                self.simulaion_outputs[wavelength] = dict()
                for resolution in self.aia.resolutions:
                    if (wavelength in self.aia.wavelengths) and (
                        resolution in self.aia.resolutions
                    ):
                        logger.info(
                            f"Multiton: Running CHIPS for {wavelength}/{resolution}"
                        )
                        disk = self.aia.datasets[wavelength][resolution]
                        self.simulaion_outputs[wavelength][
                            resolution
                        ] = self.process_CHIPS(disk)
                    else:
                        logger.error(f"No entry for {wavelength} and {resolution}!!!")
        return

    def process_CHIPS(
        self,
        disk,
    ) -> None:
        """This method runs the CHIPS algorithm for a selected `chips.fetch.SolarDisk` object.

        Attributes:
            disk (chips.fetch.SolarDisk): Solar disk to run CHIPS algorithm.

        Returns:
            Method returns None.
        """
        self.extract_solar_masks(disk)
        self.run_filters(disk)
        self.extract_histogram(disk)
        # self.extract_histograms(disk)
        # self.extract_sliding_histograms(disk)
        self.extract_CHs_CHBs(disk)
        self.plot_diagonestics(disk)
        return

    def extract_solar_masks(self, disk) -> None:
        """This method extract the solar disk mask, using method described in this [Section](../../tutorial/workings/).

        Attributes:
            disk (chips.fetch.SolarDisk): Solar disk to run CHIPS algorithm.

        Returns:
            Method returns None.
        """
        logger.info(f"Create solar mask for {disk.wavelength}/{disk.resolution}")
        if not hasattr(disk, "solar_mask"):
            n_mask = np.zeros_like(disk.raw.data) * np.nan
            cv2.circle(
                n_mask,
                (int(disk.resolution / 2), int(disk.resolution / 2)),
                disk.pixel_radius,
                255,
                -1,
            )
            n_mask[n_mask > 1] = 1.0
            i_mask = np.copy(n_mask)
            i_mask[np.isnan(i_mask)] = 0.0
            i_mask[i_mask > 1] = 1.0
            disk.set_value(
                "solar_mask", Namespace(**dict(n_mask=n_mask, i_mask=i_mask))
            )
        return

    def run_filters(self, disk) -> None:
        """This method runs the Gaussian filter on solar disk, using method described in this [Section](../../tutorial/workings/).

        Attributes:
            disk (chips.fetch.SolarDisk): Solar disk to run CHIPS algorithm.

        Returns:
            Method returns None.
        """
        logger.info(f"Running solar filters for {disk.wavelength}/{disk.resolution}")
        if not hasattr(disk, "solar_filter"):
            solar_disk = disk.normalized.data * disk.solar_mask.i_mask
            filt_disk = disk.solar_mask.i_mask * signal.medfilt2d(
                disk.normalized.data, self.medfilt_kernel
            )
            disk.set_value(
                "solar_filter",
                Namespace(**dict(solar_disk=solar_disk, filt_disk=filt_disk)),
            )
        return

    def extract_histogram(self, disk) -> None:
        """This method extract Otsu's threshold, using method described in this [Section](../../tutorial/workings/).

        Attributes:
            disk (chips.fetch.SolarDisk): Solar disk to run CHIPS algorithm.

        Returns:
            Method returns None.
        """
        logger.info(f"Extract solar histogram {disk.wavelength}/{disk.resolution}")
        if not hasattr(disk, "histogram"):
            h_data = disk.solar_filter.filt_disk * disk.solar_mask.n_mask
            h, be = np.histogram(
                h_data.ravel()[~np.isnan(h_data.ravel())],
                bins=self.h_bins,
                density=True,
            )
            if self.h_thresh is None:
                self.h_thresh = np.max(h) / self.ht_peak_ratio
            peak_indexs, _ = signal.find_peaks(h, height=self.h_thresh)
            bc = be[:-1] + np.diff(be)
            peaks = [bc[p] for p in peak_indexs]
            disk.set_value(
                "histogram",
                Namespace(
                    **dict(
                        peaks=peaks,
                        hbase=h,
                        bound=bc,
                        h_thresh=self.h_thresh,
                        h_bins=self.h_bins,
                    )
                ),
            )
        return

    def extract_histograms(self, disk) -> None:
        """Method extracting Otsu's thresolds by splitting solar disk into different windows (windows are estimated using `xsplit`, `ysplit`).

        Attributes:
            disk (chips.fetch.SolarDisk): Solar disk information for each wavelenght bands.

        Returns:
            Method returns None.
        """
        logger.info(
            f"Extract solar histograms for different regions {disk.wavelength}/{disk.resolution}"
        )
        if (
            not hasattr(disk, "histograms")
            and (self.hist_xsplit is not None)
            and (self.hist_ysplit is not None)
        ):
            histograms, peaks, k = {}, [], 0
            filt_disk, n_mask = (disk.solar_filter.filt_disk, disk.solar_mask.n_mask)
            xunit, yunit = (
                int(disk.resolution / self.hist_xsplit),
                int(disk.resolution / self.hist_ysplit),
            )
            for x in range(self.hist_xsplit):
                for y in range(self.hist_ysplit):
                    r_disk_data = (
                        filt_disk[
                            x * xunit : (x + 1) * xunit, y * yunit : (y + 1) * xunit
                        ]
                        * n_mask[
                            x * xunit : (x + 1) * xunit, y * yunit : (y + 1) * xunit
                        ]
                    ).ravel()
                    h, be = np.histogram(
                        r_disk_data[~np.isnan(r_disk_data.ravel())],
                        bins=self.h_bins,
                        density=True,
                    )
                    if self.h_thresh is None:
                        self.h_thresh = np.max(h) / self.ht_peak_ratio
                    peaks, _ = signal.find_peaks(h, height=self.h_thresh)
                    bc = be[:-1] + np.diff(be)
                    if len(peaks) > 1:
                        r_peaks.extend(peaks[:2])
                    regions[k] = Namespace(
                        **dict(
                            peaks=peaks,
                            hbase=h,
                            bound=bc,
                            data=d,
                            h_thresh=self.h_thresh,
                            h_bins=self.h_bins,
                        )
                    )
                    k += 1
            disk.set_value(
                "histograms",
                Namespace(**dict(reg=regions, fpeak=r_peaks, keys=range(k))),
            )
        return

    def extract_sliding_histograms(self, disk) -> None:
        """Method extracting Otsu's thresolds by sliding a window (size determined by `xsplit`, `ysplit`) through the solar disk.

        Attributes:
            disk (chips.fetch.SolarDisk): Solar disk information for each wavelenght bands.

        Returns:
            Method returns None.
        """
        logger.info(
            f"Extract solar histograms for different regions {disk.wavelength}/{disk.resolution}"
        )
        return

    def extract_CHs_CHBs(
        self, 
        disk
    ) -> None:
        """Method extracting coronal hole and boundaries using method described in this [Section](../../tutorial/workings/).

        Attributes:
            disk (chips.fetch.SolarDisk): Solar disk information for each wavelenght bands.

        Returns:
            Method returns None.
        """
        logger.info(
            f"Extract CHs and CHBs for different regions {disk.wavelength}/{disk.resolution}"
        )
        threshold_range = np.arange(self.threshold_range[0], self.threshold_range[1])
        if not hasattr(disk, "solar_ch_regions"):
            if len(disk.histogram.peaks) > 0:
                limit_0 = disk.histogram.peaks[0]
                limits = np.round(limit_0 + threshold_range, 4)
                data = disk.solar_filter.filt_disk * disk.solar_mask.n_mask
                dtmp_map = {}
                for lim in limits:
                    tmp_data = np.copy(data)
                    nd_mask = np.zeros_like(data) * np.nan
                    cv2.circle(
                        nd_mask,
                        (int(disk.resolution / 2), int(disk.resolution / 2)),
                        disk.pixel_radius,
                        255,
                        -1,
                    )
                    nd_mask[nd_mask > 0] = 1
                    tmp_data = nd_mask * tmp_data
                    tmp_data[tmp_data <= lim] = -1
                    tmp_data[tmp_data > lim] = 0
                    tmp_data[tmp_data == -1] = 1
                    tmp_data[np.isnan(tmp_data)] = 0
                    p = self.calculate_prob(np.copy(data).ravel(), [lim, limit_0])
                    dtmp_map[str(lim)] = Namespace(
                        **{
                            "lim": lim,
                            "limit_0": limit_0,
                            "prob": p,
                            "map": np.copy(tmp_data),
                        }
                    )
                    logger.info(f"Estimated prob.({p}) at lim({lim})")
                disk.set_value("solar_ch_regions", Namespace(**dtmp_map))
        return

    def calculate_prob(
        self, 
        data: np.array, 
        threshold: float
    ) -> float:
        r"""Method to estimate probability for each CH region identified by CHIPS.

        Attributes:
            data (np.array): Numpy 1D array holding '.fits' level intensity (I) dataset. 
            threshold (float): Intensity thresold ($I_{th}$).

        Returns:
            p (float): Probability [$\theta$] of each region estimated using Beta function.
        """
        data = data[~np.isnan(data)]
        data = data[data <= thresholds[0]]
        ps = 1.0 / (1.0 + np.exp(data - thresholds[1]).round(4))
        h, e = np.histogram(ps, bins=np.linspace(0, 1, 21))
        b = (e + np.diff(e)[0] / 2)[:-1]
        p = np.sum(b[b > self.porb_threshold] * h[b > self.porb_threshold]) / np.sum(
            b * h
        )
        return np.round(p, 4)

    def plot_diagonestics(
        self,
        disk,
        figsize=(6, 6),
        dpi=240,
        nrows=2,
        ncols=2,
        prob_lower_lim: float=0.,
    ) -> None:
        """Method to create diagonestics plots showing different steps of CHIPS. 
        Expected file formats png, bmp, jpg, pdf, etc.

        Attributes:
            disk (chips.fetch.SolarDisk): Solar disk object that holds all information for drawing. 
            figsize (Tuple): Figure size (width, height)
            dpi (int): Dots per linear inch.
            nrows (int): Number of axis rows in a figure palete.
            ncols (int): Number of axis colums in a figure palete.
            prob_lower_lim (float): Minimum limit of the color bar.
        
        Returns:
            Method returns None.
        """
        cp = ChipsPlotter(
            disk,
            figsize,
            dpi,
            nrows,
            ncols,
        )
        cp.create_diagonestics_plots(
            self.folder + f"/diagonestics_{disk.wavelength}_{disk.resolution}.png",
            prob_lower_lim=prob_lower_lim,
        )
        cp.create_output_stack(
            fname=self.folder + f"/ouputstack_{disk.wavelength}_{disk.resolution}.png"
        )
        return
