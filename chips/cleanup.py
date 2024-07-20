"""
    cleanup.py: Coronal Hole Detection Cleanup Pipeline, CH/Fl
        Idea and code originally developed by Tadhg Garton and published in Garton et al. (2018).
    
    Description:
        This module is part of the CHIPS framework. It is designed to remove filaments that 
        are erroneously labeled as coronal holes by creating a map of coronal hole candidates 
        based on the CHIMERA detection approach. These candidates can later be overlaid with 
        CHIPS detections to improve the results.

    Caveats:
        No magnetic polarity check and cleanup of small structures have been included in this 
        code yet.
"""

__author__ = "Reiss, M.(NASA CCMC)"
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Reiss M.; Chakrabrty, S."
__email__ = "martin.a.reiss@outlook.com; shibaji7@vt.edu"
__status__ = "Research"

import numpy as np
from typing import Tuple
from argparse import Namespace
import sunpy
from skimage.transform import resize
import cv2

from chips.fetch import RegisterAIA
from chips.plots import ImagePalette

class CleanFl(object):
    r"""An object class that runs the CHIPS/Fl clean up algorithm step-by-step with a set of 
        input parameters.

    Attributes:
        aia (chips.fetch.RegisterAIA): AIA/Map object that holds all the information on `List[chips.fetch.SolarDisk]`.
        base_folder (str): Base folder to store the processed figures and datasets.
        params (dict): Parameters to optimize the cleanups.
        resolution (int): Image resolution.
    """

    def __init__(
            self,
            aia: RegisterAIA,
            resolution: int = 4096,
            params: dict = dict(
                clip_negative=0,
                clip_211_log = [0.8, 2.7],
                clip_193_log = [1.4, 3.0],
                clip_171_log = [1.2, 3.9],
                bmmix_value = 0.6357,
                bmhot_value = 0.7,
                bmcool_value = 1.5102,
            )
    ):
        """Initialization method"""
        self.aia = aia
        self.base_folder = self.aia.save_dir
        self.params = Namespace(**params)
        self.resolution = resolution
        return
    
    def create_coronal_hole_candidates(
            self,
            smap193: sunpy.map.Map = None, 
            smap171: sunpy.map.Map = None, 
            smap211: sunpy.map.Map = None, 
            disk_mask: np.array = None,
            resolution: int = None
        ) -> Tuple:
        """
        Create coronal hole candidates by thresholding SDO/AIA images from three different wavelengths.
        
        Attributes:
            smap193 (sunpy.map.Map): SunPy map for 193 Å data (level 1).
            smap171 (sunpy.map.Map): SunPy map for 171 Å data (level 1).
            smap211 (sunpy.map.Map): SunPy map for 211 Å data (level 1).
            disk_mask (np.array): Solar disk mask.
            resolution (int): Desired resolution for the output bitmaps.
        
        Returns:
            Coronal hole candidate bitmap and individual bitmaps (cand, bmmix, bmhot, bmcool, disk_mask).
        """

        resolution = resolution if resolution else self.resolution
        smap193 = smap193 if smap193 else self.aia.datasets[193][resolution].normalized
        smap171 = smap171 if smap171 else self.aia.datasets[171][resolution].normalized
        smap211 = smap211 if smap211 else self.aia.datasets[211][resolution].normalized

        if disk_mask is None:
            disk_mask = np.zeros((resolution, resolution))
            cv2.circle(
                disk_mask,
                (int(resolution / 2), int(resolution / 2)),
                self.aia.datasets[193][self.resolution].pixel_radius,
                255,
                -1,
            )
            disk_mask[disk_mask > 1] = 1.0

        def clip_limits(data, lims):
            """
            Cliping the dataset based on limits
            """
            return np.clip(data, lims[0], lims[1])
        
        def scale_limits(data, lims, max=255):
            """
            Scaled the dataset based on limits
            """
            return np.array(255 * (data-lims[0]) / (lims[1]-lims[0]), dtype=np.float32)

        # Rescale to ensure all maps have the same size
        data171_rescaled, data193_rescaled, data211_rescaled = (
            resize(
                smap171.data, (resolution, resolution), 
                mode="reflect", anti_aliasing=True
            ),
            resize(
                smap193.data, (resolution, resolution),
                mode="reflect", anti_aliasing=True
            ),
            resize(
                smap211.data, (resolution, resolution), 
                mode="reflect", anti_aliasing=True
            )
        )

        # Remove negative data values
        data171_rescaled[data171_rescaled <= self.params.clip_negative] = 0
        data193_rescaled[data193_rescaled <= self.params.clip_negative] = 0
        data211_rescaled[data211_rescaled <= self.params.clip_negative] = 0

        # Initialize bitmaps
        bmcool = np.zeros((resolution, resolution), dtype=np.float32)
        bmmix, bmhot = np.copy(bmcool), np.copy(bmcool)

        # Create multi-wavelength image for contours
        with np.errstate(divide="ignore"):
            log_data211 = np.log10(data211_rescaled)
            log_data193 = np.log10(data193_rescaled)
            log_data171 = np.log10(data171_rescaled)

        cliped_log_data211 = clip_limits(log_data211, self.params.clip_211_log)
        cliped_log_data193 = clip_limits(log_data193, self.params.clip_193_log)
        cliped_log_data171 = clip_limits(log_data171, self.params.clip_171_log)

        scaled_log_data211 = scale_limits(cliped_log_data211, self.params.clip_211_log)
        scaled_log_data193 = scale_limits(cliped_log_data193, self.params.clip_193_log)
        scaled_log_data171 = scale_limits(cliped_log_data171, self.params.clip_171_log)
        
        # Create segmented bitmasks
        with np.errstate(divide="ignore", invalid="ignore"):
            bmmix[scaled_log_data171 / scaled_log_data211 >= (np.mean(data171_rescaled) * self.params.bmmix_value / np.mean(data211_rescaled))] = 1
            bmhot[scaled_log_data211 + scaled_log_data193 < (self.params.bmhot_value * (np.mean(data193_rescaled) + np.mean(data211_rescaled)))] = 1
            bmcool[scaled_log_data171 / scaled_log_data193 >= (np.mean(data171_rescaled) * self.params.bmcool_value / np.mean(data193_rescaled))] = 1

        # Logical conjunction of 3 segmentations
        candidate_bitmap = bmcool * bmmix * bmhot
        self.candidate_bitmap, self.bmmix, self.bmhot, self.bmcool = (
            candidate_bitmap*disk_mask, bmmix*disk_mask, bmhot*disk_mask, bmcool*disk_mask
        )
        return candidate_bitmap, bmmix, bmhot, bmcool, disk_mask 
    
    def produce_summary_plots(
            self,
            file_name: str = None,
            figsize: Tuple = (6, 6),
            dpi: int = 240,
            nrows: int = 2,
            ncols: int = 2,
        ) -> None:
        """
        This method create a 2X2 summary plots

        Attributes:

        
        Returns:
            Method returns None
        """
        date = self.aia.date
        def draw_panel(ax, data, title):
            """ Draw each individual panels
            """
            ip.__circle__(
                ax, self.aia.datasets[193][self.resolution].pixel_radius, 
                self.resolution, color="m"
            )
            ax.set_xlim(0, 4097)
            ax.set_ylim(0, 4097)
            ax.imshow(data, cmap="gray", origin="lower")
            ax.text(0.1, 0.95, title, ha="left", va="center", fontdict=dict(color="y", size=12), transform=ax.transAxes)
            return ax
        
        ip = ImagePalette(
            figsize=figsize,
            dpi=dpi,
            nrows=nrows,
            ncols=ncols,
        )
        ax = draw_panel(ip.__axis__(), self.bmmix, r"211 $\AA$ / 171 $\AA$")
        ax.text(
            0.1, 1.05, date.strftime("%Y-%m-%d %H:%M"), 
            ha="left", va="center", fontdict=dict(color="k", size=12), 
            transform=ax.transAxes
        )
        draw_panel(ip.__axis__(), self.bmhot, r"211 $\AA$ / 193 $\AA$")
        ax = draw_panel(ip.__axis__(axis_off=False), self.bmcool, r"193 $\AA$ / 171 $\AA$")
        ax.set_xticks(np.arange(0, 4097, 1024))
        ax.set_yticks(np.arange(0, 4097, 1024))
        ax.set_xticklabels([-2048, -1024, 0, 1024, 2048])
        ax.set_yticklabels([-2048, -1024, 0, 1024, 2048])
        ax.set_xlabel("pixels")
        ax.set_ylabel("pixels")
        draw_panel(ip.__axis__(), self.candidate_bitmap, "Logical Conjunction")
        file_name = file_name if file_name else "filament_cleanups.png"
        ip.save(self.base_folder + file_name)
        return
