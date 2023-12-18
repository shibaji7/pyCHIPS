"""
    syn_chips.py: Module is used to implement edge detection techniques using thresholding for synoptic maps.
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
import numpy as np
from loguru import logger
from plots import ChipsPlotter
from scipy import signal

class SynopticChips(object):
    r"""An object class that runs the CHIPS algorithm on synoptic maps step-by-step with a set of input parameters.

    Attributes:
        synoptic_map (chips.fetch.SynopticMap): Map object that holds all the information on `chips.fetch.SynopticMap`.
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
        synoptic_map,
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
        """Initialization method"""
        self.synoptic_map = synoptic_map
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

    def run_CHIPS(self) -> None:
        """This method selects the `chips.fetch.SynopticMap` objects and runs the CHIPS algorithm sequentially.

        Attributes:

        Returns:
            Method returns None.
        """
        self.run_filters(self.synoptic_map)
        self.extract_histogram(self.synoptic_map)
        # self.extract_histograms(self.synoptic_map)
        # self.extract_sliding_histograms(self.synoptic_map)
        self.extract_CHs_CHBs(self.synoptic_map)
        self.plot_diagonestics(self.synoptic_map)
        return

    def run_filters(self, synoptic_map) -> None:
        """This method runs the Gaussian filter on solar Synoptic Maps, using method described in this [Section](../../tutorial/workings/).

        Attributes:
            synoptic_map (chips.fetch.SynopticMap): Solar synoptic Maps to run CHIPS algorithm.

        Returns:
            Method returns None.
        """
        logger.info(f"Running solar filters for {synoptic_map.wavelength}")
        if not hasattr(synoptic_map, "solar_filter"):
            solar_filter = signal.medfilt2d(
                synoptic_map.raw.data, self.medfilt_kernel
            )
            synoptic_map.set_value("solar_filter", solar_filter)
        return

    def extract_histogram(self, synoptic_map) -> None:
        """This method extract Otsu's threshold, using method described in this [Section](../../tutorial/workings/).

        Attributes:
            synoptic_map (chips.fetch.SynopticMap): Solar synoptic Maps to run CHIPS algorithm.

        Returns:
            Method returns None.
        """
        logger.info(f"Extract solar histogram {synoptic_map.wavelength}")
        if not hasattr(synoptic_map, "histogram"):
            h_data = synoptic_map.solar_filter
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
            synoptic_map.set_value(
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

    def extract_histograms(self, synoptic_map) -> None:
        """Method extracting Otsu's thresolds by splitting solar Synoptic Map into different windows (windows are estimated using `xsplit`, `ysplit`).

        Attributes:
            synoptic_map (chips.fetch.SynopticMap): Solar synoptic Maps to run CHIPS algorithm.

        Returns:
            Method returns None.
        """
        logger.info(
            f"Extract solar histograms for different regions {synoptic_map.wavelength}"
        )
        if (
            not hasattr(synoptic_map, "histograms")
            and (self.hist_xsplit is not None)
            and (self.hist_ysplit is not None)
        ):
            histograms, peaks, k = {}, [], 0
            solar_filter = synoptic_map.solar_filter
            xunit, yunit = (
                int(synoptic_map.resolution / self.hist_xsplit),
                int(synoptic_map.resolution / self.hist_ysplit),
            )
            for x in range(self.hist_xsplit):
                for y in range(self.hist_ysplit):
                    r_data = (
                        solar_filter[
                            x * xunit : (x + 1) * xunit, y * yunit : (y + 1) * xunit
                        ]
                    ).ravel()
                    h, be = np.histogram(
                        r_data[~np.isnan(r_data.ravel())],
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
            synoptic_map.set_value(
                "histograms",
                Namespace(**dict(reg=regions, fpeak=r_peaks, keys=range(k))),
            )
        return

    def extract_sliding_histograms(self, synoptic_map) -> None:
        """Method extracting Otsu's thresolds by sliding a window (size determined by `xsplit`, `ysplit`) through the solar Synoptic Map.

        Attributes:
            synoptic_map (chips.fetch.SynopticMap): Solar synoptic Maps to run CHIPS algorithm.

        Returns:
            Method returns None.
        """
        logger.info(
            f"Extract solar histograms for different regions {synoptic_map.wavelength}"
        )
        return

    def extract_CHs_CHBs(self, synoptic_map) -> None:
        """Method extracting coronal hole and boundaries using method described in this [Section](../../tutorial/workings/).

        Attributes:
            synoptic_map (chips.fetch.SynopticMap): Solar synoptic Maps to run CHIPS algorithm.

        Returns:
            Method returns None.
        """
        logger.info(
            f"Extract CHs and CHBs for different regions {synoptic_map.wavelength}"
        )
        threshold_range = np.arange(self.threshold_range[0], self.threshold_range[1])
        if not hasattr(synoptic_map, "solar_ch_regions"):
            if len(synoptic_map.histogram.peaks) > 0:
                limit_0 = synoptic_map.histogram.peaks[0]
                limits = np.round(limit_0 + threshold_range, 4)
                data = synoptic_map.solar_filter
                dtmp_map = {}
                for lim in limits:
                    tmp_data = np.copy(data)
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
                synoptic_map.set_value("solar_ch_regions", Namespace(**dtmp_map))
        return

    def calculate_prob(self, data: np.array, threshold: float) -> float:
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
        synoptic_map,
        figsize=(6, 6),
        dpi=240,
        nrows=2,
        ncols=2,
        prob_lower_lim: float = 0.0,
    ) -> None:
        """Method to create diagonestics plots showing different steps of CHIPS.
        Expected file formats png, bmp, jpg, pdf, etc.

        Attributes:
            synoptic_map (chips.fetch.SynopticMap): Solar Synoptic Map object that holds all information for drawing.
            figsize (Tuple): Figure size (width, height)
            dpi (int): Dots per linear inch.
            nrows (int): Number of axis rows in a figure palete.
            ncols (int): Number of axis colums in a figure palete.
            prob_lower_lim (float): Minimum limit of the color bar.

        Returns:
            Method returns None.
        """
        cp = ChipsPlotter(
            synoptic_map,
            figsize,
            dpi,
            nrows,
            ncols,
        )
        cp.create_diagonestics_plots(
            self.folder + f"/synoptic_diagonestics_{synoptic_map.wavelength}.png",
            prob_lower_lim=prob_lower_lim,
        )
        cp.create_output_stack(
            fname=self.folder + f"/synoptic_ouputstack_{synoptic_map.wavelength}.png"
        )
        return