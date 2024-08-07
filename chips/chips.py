"""
    chips.py: Module is used to implement edge detection techniques using thresholding.
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
from typing import Dict, List, Tuple

import cv2
import numpy as np
from loguru import logger
from netCDF4 import Dataset
from scipy import signal

from chips.cleanup import CleanFl
from chips.fetch import RegisterAIA
from chips.plots import ChipsPlotter


class Chips(object):
    r"""An object class that runs the CHIPS algorithm step-by-step with a set of input parameters.

    Attributes:
        aia (chips.fetch.RegisterAIA): AIA/Map object that holds all the information on `List[chips.fetch.SolarDisk]`.
        base_folder (str): Base folder to store the processed figures and datasets.
        medfilt_kernel (int): Median filtering Gaussian kernel size (odd number).
        h_bins (int): Number of bins in histogram while running Otsu's method.
        h_thresh (float): Otsu's thresold (computed inside if not given, $I_{th}$).
        ht_peak_ratio (int): Needed to cpmpute threhold while computing Otsu's method.
        hist_xsplit (int): Needed to compute thresholds by splitted image (number of splitted along the image width).
        hist_ysplit (int): Needed to compute thresholds by splitted image (number of splitted along the image height).
        threshold_range (List[float]): List of thresholds from `h_thresh` to run CHIPS.
        porb_threshold (int): This is $x_{\tau}$.
        area_threshold (float): Percentage (0-1) area thresholds to remove small structures.
        run_fl_cleanup (bool): Run the filament cleanups.
    """

    def __init__(
        self,
        aia,
        base_folder: str = "tmp/chips_dataset/",
        medfilt_kernel: int = 51,
        h_bins: int = 500,
        h_thresh: float = None,
        ht_peak_ratio: int = 5,
        hist_xsplit: int = 2,
        hist_ysplit: int = 2,
        threshold_range: List[float] = [0, 20],
        porb_threshold: float = 0.8,
        area_threshold: float = 1e-3,
        run_fl_cleanup: bool = False,
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
        self.area_threshold = area_threshold
        self.run_fl_cleanup = run_fl_cleanup
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

    def clear_run_results(self, disk):
        """This method clears the model outputs.

        Attributes:

        Returns:
            Method returns None.
        """
        if hasattr(disk, "solar_mask"):
            delattr(disk, "solar_mask")
        if hasattr(disk, "solar_filter"):
            delattr(disk, "solar_filter")
        if hasattr(disk, "histogram"):
            delattr(disk, "histogram")
        if hasattr(disk, "histograms"):
            delattr(disk, "histograms")
        if hasattr(disk, "solar_ch_regions"):
            delattr(disk, "solar_ch_regions")
        return

    def run_CHIPS(
        self,
        wavelength: int = None,
        resolution: int = None,
        clear_prev_runs: bool = False,
    ) -> None:
        """This method selects the `chips.fetch.SolarDisk`(s) objects and runs the CHIPS algorithm sequentially.

        Attributes:
            wavelength (int): Wave length of the disk image [171/193/211].
            resolution (int): Resolution of the image to work on [4096].
            clear_prev_runs (bool): Clearing the outputs from previou run.

        Returns:
            Method returns None.
        """
        if self.run_fl_cleanup:
            self.run_filament_cleanup()
        resolution = self.aia.resolution if resolution is None else resolution
        if (wavelength is not None) and (resolution is not None):
            if (wavelength in self.aia.wavelengths) and (
                resolution == self.aia.resolution
            ):
                logger.info(f"Singleton: Running CHIPS for {wavelength}/{resolution}")
                disk = self.aia.datasets[wavelength][resolution]
                self.process_CHIPS(disk, clear_prev_runs)
            else:
                logger.error(f"No entry for {wavelength} and {resolution}!!!")
        else:
            for wavelength in self.aia.wavelengths:
                if (wavelength in self.aia.wavelengths) and (
                    resolution is self.aia.resolution
                ):
                    logger.info(
                        f"Multiton: Running CHIPS for {wavelength}/{resolution}"
                    )
                    disk = self.aia.datasets[wavelength][resolution]
                    self.process_CHIPS(disk, clear_prev_runs)
                else:
                    logger.error(f"No entry for {wavelength} and {resolution}!!!")
        return

    def process_CHIPS(self, disk, clear_prev_runs) -> None:
        """This method runs the CHIPS algorithm for a selected `chips.fetch.SolarDisk` object.

        Attributes:
            disk (chips.fetch.SolarDisk): Solar disk to run CHIPS algorithm.
            clear_prev_runs (bool): Clearing the outputs from previou run.

        Returns:
            Method returns None.
        """
        if clear_prev_runs:
            self.clear_run_results(disk)
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
            self.disk_area = np.pi * disk.pixel_radius**2
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
                        hbins=be,
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
        if (
            not hasattr(disk, "histograms")
            and (self.hist_xsplit is not None)
            and (self.hist_ysplit is not None)
        ):
            pass
        return

    def extract_CHs_CHBs(self, disk) -> None:
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
                    if self.run_fl_cleanup:
                        tmp_data = tmp_data * self.cf.candidate_bitmap
                    p = self.calculate_prob(np.copy(data).ravel(), [lim, limit_0])
                    ##############################################################
                    # Calculate region by CV2 find contour function
                    ##############################################################
                    contours, hierarchy = cv2.findContours(
                        tmp_data.astype(np.uint8),
                        cv2.RETR_TREE,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    contours, hierarchy, dxmap = self.clean_small_scale_structures(
                        contours, hierarchy, np.zeros_like(tmp_data)
                    )
                    dtmp_map[str(lim)] = Namespace(
                        **{
                            "lim": lim,
                            "limit_0": limit_0,
                            "prob": p,
                            "map": dxmap,
                            "contours": contours,
                            "hierarchy": hierarchy,
                            "contour_ids": [f"CH_{ix}" for ix in range(len(contours))],
                        }
                    )
                    logger.info(f"Estimated prob.({p}) at lim({lim})")
                disk.set_value("solar_ch_regions", Namespace(**dtmp_map))
        return

    def clean_small_scale_structures(
        self,
        contours: List[np.array],
        hierarchy: np.array,
        data: np.array,
    ) -> tuple:
        r"""Remove small scale structures from the contour list.

        Attributes:
            contours (List[np.array]): List containing all the countours
            hierarchy (np.array): Hierarchy of the contours
            data (np.array): Data array to create maps

        Returns:
            dataset (tuple): Tuple containing modified contours and hierarchy
        """
        dxmap = np.copy(data)
        new_contours, new_hierarchy = list(), list()
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area / self.disk_area >= self.area_threshold:
                new_contours.append(cnt)
                if hierarchy[0][i][3] < 0:
                    cv2.drawContours(dxmap, [cnt], -1, (255, 255, 255), -1)
                new_hierarchy.append(hierarchy[0][i])
        return (new_contours, np.array(new_hierarchy), dxmap / 255)

    def calculate_prob(self, data: np.array, thresholds: List[float]) -> float:
        r"""Method to estimate probability for each CH region identified by CHIPS.

        Attributes:
            data (np.array): Numpy 1D array holding '.fits' level intensity (I) dataset.
            thresholds (List[float]): Intensity thresolds ($I_{th}$).

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
        dpi=240,
        prob_lower_lim: float = 0.0,
        vert: np.array = None,
    ) -> None:
        """Method to create diagonestics plots showing different steps of CHIPS.
        Expected file formats png, bmp, jpg, pdf, etc.

        Attributes:
            disk (chips.fetch.SolarDisk): Solar disk object that holds all information for drawing.
            dpi (int): Dots per linear inch.
            prob_lower_lim (float): Minimum limit of the color bar.
            vert (np.array): Vertices of rectange to zoom in a specific region [lower left, upper right points].

        Returns:
            Method returns None.
        """
        text = r"$\kappa={%d}$, $h_{bins}={%d}$" % (self.medfilt_kernel, self.h_bins)
        parameter_details = dict(xloc=0.8, yloc=1.01, text=text)
        cp = ChipsPlotter(disk, dpi=dpi, parameter_details=parameter_details)
        cp.create_diagonestics_plots(
            self.folder
            + f"/diagonestics_solid_{disk.wavelength}_{disk.resolution}.png",
            prob_lower_lim=prob_lower_lim,
            figsize=(9, 3),
            nrows=1,
            ncols=3,
            vert=vert,
        )
        cp.create_diagonestics_plots(
            self.folder
            + f"/diagonestics_contour_{disk.wavelength}_{disk.resolution}.png",
            prob_lower_lim=prob_lower_lim,
            figsize=(9, 3),
            nrows=1,
            ncols=3,
            solid_fill=False,
            vert=vert,
        )
        cp.create_output_stack(
            fname=self.folder
            + f"/ouputstack_solid_{disk.wavelength}_{disk.resolution}.png",
            figsize=(6, 6),
            nrows=2,
            ncols=2,
            vert=vert,
        )
        cp.create_output_stack(
            fname=self.folder
            + f"/ouputstack_contour_{disk.wavelength}_{disk.resolution}.png",
            figsize=(6, 6),
            nrows=2,
            ncols=2,
            solid_fill=False,
            vert=vert,
        )
        return

    def to_netcdf(
        self, wavelength: int, resolution: int, file_name: str = None
    ) -> None:
        """Method to save the CHIPS object to netCDF files.
        Expected file formats netCDF.

        Attributes:
            wavelength (int): Wave length of the disk image [171/193/211].
            resolution (int): Resolution of the image to work on [4096].
            file_name (str): Name of the file (.nc).

        Returns:
            Method returns None.
        """

        def create_dataset(grp, dims, name, dtype, value):
            ds = grp.createVariable(name, dtype, dims)
            ds[:] = value
            return

        def create_group_with_dims(grp, sub_grp_name, dims):
            sub_grp = grp.createGroup(sub_grp_name)
            for dim in dims:
                sub_grp.createDimension(dim[0], dim[1])
            return sub_grp

        disk = self.aia.datasets[wavelength][resolution]
        file_name = (
            file_name
            if file_name
            else f"{disk.date.strftime('%Y-%m-%d-%H-%M')}_{wavelength}A_{resolution}.nc"
        )
        file = self.folder + f"/{file_name}"
        logger.info(f"Save to file: {file}")
        nc = Dataset(file, "w", format="NETCDF4")
        fits = create_group_with_dims(
            nc, "fits", [("xarcs", resolution), ("yarcs", resolution)]
        )
        create_dataset(fits, ("xarcs", "yarcs"), "fitdata", "f4", disk.normalized.data)
        if hasattr(disk, "solar_filter"):
            filter = create_group_with_dims(
                nc, "filter", [("xarcs", resolution), ("yarcs", resolution)]
            )
            create_dataset(
                filter,
                ("xarcs", "yarcs"),
                "solar_disk",
                "f4",
                disk.solar_filter.solar_disk,
            )
            create_dataset(
                filter,
                ("xarcs", "yarcs"),
                "filt_disk",
                "f4",
                disk.solar_filter.filt_disk,
            )

        if hasattr(disk, "solar_ch_regions"):
            L, keys = (
                len(disk.solar_ch_regions.__dict__.keys()),
                list(disk.solar_ch_regions.__dict__.keys()),
            )
            ch_regions = create_group_with_dims(
                nc,
                "ch_regions",
                [("xarcs", resolution), ("yarcs", resolution), ("probs", L)],
            )
            data, probs = np.zeros((resolution, resolution, L)), []
            for i, k in enumerate(keys):
                region = getattr(disk.solar_ch_regions, str(k))
                data[:, :, i] = region.map
                probs.append(region.prob)
            create_dataset(
                ch_regions, ("xarcs", "yarcs", "probs"), "ch_regions", "f4", data
            )
            create_dataset(ch_regions, ("probs"), "probs", "f4", probs)
        nc.close()
        return

    def compute_similarity_measures(
        self,
        map: np.ndarray,
        map0: np.ndarray,
        measure: str = "Hu",
    ) -> Dict:
        """This method compute similarity matrices (cosine, Hu)

        Attributes:
            map (np.ndarray): An nD(typically 2D) array containing the map for comparison.
            map0 (np.ndarray): Baseline solar map to compare againts.
            measure (str): Similarity measure name [cos/Hu]

        Returns:
            Method returns dictionary of measures.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        measures = {"cosine_sim": np.nan}
        cosine_sim = []
        for i in range(map.shape[0]):
            if np.sum(map[i, :]) > 0:
                a, b = (np.array([list(map[i, :])]), np.array([list(map0[i, :])]))
                cosine_sim.append(cosine_similarity(a, b)[0, 0])
        measures["cosine_sim"] = np.nanmean(cosine_sim)
        return measures

    def run_filament_cleanup(
        self,
        params: dict = dict(
            clip_negative=0,
            clip_211_log=[0.8, 2.7],
            clip_193_log=[1.4, 3.0],
            clip_171_log=[1.2, 3.9],
            bmmix_value=0.6357,
            bmhot_value=0.7,
            bmcool_value=1.5102,
        ),
        file_name: str = None,
        figsize: Tuple = (6, 6),
        dpi: int = 240,
        nrows: int = 2,
        ncols: int = 2,
    ) -> None:
        """
        This method clears all possible filaments.

        Attributes:
            params (dict): Parameters to optimize the cleanups.
            resolution (int): Image resolution.
            figsize (Tuple): Figure size (width, height)
            dpi (int): Dots per linear inch.
            nrows (int): Number of axis rows in a figure palete.
            ncols (int): Number of axis colums in a figure palete.

        Returns:
            Method returns none
        """
        logger.info("Into the filament Cleanups!")
        self.cf = CleanFl(
            RegisterAIA(
                self.aia.date,
                [171, 193, 211],
                self.aia.resolution,
                apply_psf=False,
                local_file="sunpy/data/aia_lev1_{wavelength}a_{date_str}*.fits",
            ),
            resolution=self.aia.resolution,
            params=params,
        )
        self.cf.create_coronal_hole_candidates()
        self.cf.produce_summary_plots(
            file_name=file_name,
            figsize=figsize,
            dpi=dpi,
            nrows=nrows,
            ncols=ncols,
        )
        return
