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
        """Initialization method"""
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