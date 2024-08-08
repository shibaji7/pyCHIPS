"""
    plots.py: Module is used to hold all helper functions.
"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
import sunpy.visualization.colormaps as cm
from loguru import logger
import cv2


class Annotation(object):
    """An object class that holds annotation details that put on a disk image. The attibutes
    of this class is similar to `maplotlib.pyplot.ax.text()`.

    Attributes:
        txt (str): Text to be annotated.
        xloc (float): X-location of the text.
        yloc (float): Y-location of the text.
        ha (str): Horizontal Alignment of the text.
        va (str): Vertical Alignment of the text.
        fontdict (dict): Font dictionary containg text styles.
        rotation (float): Text rotation in degress (0-360).
    """

    def __init__(
        self,
        txt: str,
        xloc: float,
        yloc: float,
        ha: str,
        va: str,
        fontdict: dict = {"color": "k", "size": 10},
        rotation: float = 0,
    ) -> None:
        self.txt = txt
        self.xloc = xloc
        self.yloc = yloc
        self.ha = ha
        self.va = va
        self.rotation = rotation
        self.fontdict = fontdict
        return


class ImagePalette(object):
    """An object class that holds the figures and axis to draw.

    Attributes:
        figsize (Tuple): Figure size (width, height)
        dpi (int): Dots per linear inch.
        nrows (int): Number of axis rows in a figure palete.
        ncols (int): Number of axis colums in a figure palete.
        font_family (str): Font familly.
        sharex (str): Shareing X-axis ['all', 'row', 'col', 'none'].
        sharey (str): Shareing X-axis ['all', 'row', 'col', 'none'],
        vert (np.array): Vertices of rectange to zoom in a specific region [lower left, upper right points].
    """

    def __init__(
        self,
        figsize: Tuple = (6, 6),
        dpi: int = 240,
        nrows: int = 1,
        ncols: int = 1,
        font_family: str = "sans-serif",
        sharex: str = "all",
        sharey: str = "all",
        vert: np.array = None,
    ) -> None:
        plt.rcParams["font.family"] = font_family
        if font_family == "sans-serif":
            plt.rcParams["font.sans-serif"] = [
                "Tahoma",
                "DejaVu Sans",
                "Lucida Grande",
                "Verdana",
            ]
        self.fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            dpi=dpi,
        )
        self.vert = vert
        self.ticker = 0
        self.axes = []
        if nrows * ncols == 1:
            ax = self.axes.append(axs)
        else:
            self.axes.extend(axs.ravel())
        return

    def close(self) -> None:
        """Methods to close image palete.

        Arguments:

        Returns:
            Method returns None.
        """
        plt.close("all")
        return

    def save(
        self,
        fname: str,
        hspace: float = 0.01,
        wspace: float = 0.01,
        bbox_inches: str = "tight",
    ) -> None:
        """Methods to save the image into local system.

        Arguments:
            fname (str): File name to save the image (expected file formats png, bmp, jpg, pdf, etc).
            hspace (float): Subplot adjust height space.
            wspace (float): Subplot adjust width space.
            bbox_inches (str): Figure axis boxes.

        Returns:
            Method returns None.
        """
        self.fig.subplots_adjust(hspace=hspace, wspace=wspace)
        self.fig.savefig(fname, bbox_inches=bbox_inches)
        return

    def __axis__(self, ticker: int = None, axis_off: bool = True, disk_axis: bool = False) -> None:
        """Adding/fetching axis in the figure Paletes.

        Arguments:
            ticker (int): Figure axis number.
            axis_off (bool): Set both axis Off.
            disk_axis (bool): Set both axis to Disk-pixel axis.

        Returns:
            Method returns None.
        """
        if ticker is not None:
            ax = self.axes[ticker]
        else:
            ax = self.axes[self.ticker]
            self.ticker += 1
        if axis_off:
            ax.set_axis_off()
        if disk_axis and (not axis_off):
            ax.set_xticks(np.array([0, 1024, 2048, 3072, 4096]))
            ax.set_xticklabels([-2048, -1024, 0, 1024, 2048])
            ax.set_yticks(np.array([0, 1024, 2048, 3072, 4096]))
            ax.set_yticklabels([-2048, -1024, 0, 1024, 2048])
        if self.vert is not None and self.vert.shape[0]==2:
            ax.set_xlim(self.vert[0,0], self.vert[1,0])
            ax.set_ylim(self.vert[0,1], self.vert[1,1])
        return ax

    def __circle__(
        self,
        ax: matplotlib.axes.Axes,
        pixel_radius: int,
        resolution: int,
        color: str = "w",
    ) -> None:
        """Adding/fetching solar disk circle in the disk maps.

        Arguments:
            ax (matplotlib.axes.Axes): Figure axis.
            pixel_radius (int): Radious of the solar disk.
            resolution: Image resolution (typically 4k).
            color (str): Rim color.

        Returns:
            Method returns None.
        """
        ax.add_patch(
            plt.Circle(
                (resolution / 2, resolution / 2), pixel_radius, color=color, fill=False
            )
        )
        return

    def draw_colored_disk(
        self,
        map: sunpy.map.Map,
        pixel_radius: int,
        data: np.array = None,
        resolution: int = 4096,
        ticker: int = None,
        alpha: float = 1,
        draw_circle: bool = True,
        axis_off: bool = True,
    ) -> None:
        """Plotting colored solar disk images in the axis.

        Arguments:
            map (sunpy.map.Map): Sunpy map level dataset to plt solar disk.
            pixel_radius (int): Radious of the solar disk.
            data (np.array): 2D numpy array of dataset to plot (if not given `map.data` is plotted).
            resolution: Image resolution (typically 4k).
            ticker (int): Axis ticker.
            alpha (float): Figure transparency.
            draw_circle (bool): Draw the solar disk.
            axis_off (bool): Off the axis.

        Returns:
            Method returns None.
        """
        ax = self.__axis__(ticker, axis_off, disk_axis=not axis_off)
        data = data if data is not None else map.data
        norm = map.plot_settings["norm"]
        norm.vmin, norm.vmax = np.percentile(map.data, [30, 99.9])
        ax.imshow(
            data,
            norm=norm,
            cmap=map.plot_settings["cmap"],
            origin="lower",
            alpha=alpha,
        )
        if draw_circle:
            self.__circle__(ax, pixel_radius, resolution, color="m")
        return

    def draw_grayscale_disk(
        self,
        map: sunpy.map.Map,
        pixel_radius: int,
        data: np.array = None,
        resolution: int = 4096,
        ticker: int = None,
        alpha: float = 1,
        draw_circle: bool = True,
        axis_off: bool = True,
    ) -> None:
        """Plotting gray-scaled solar disk images in the axis.

        Arguments:
            map (sunpy.map.Map): Sunpy map level dataset to plt solar disk.
            pixel_radius (int): Radious of the solar disk.
            data (np.array): 2D numpy array of dataset to plot (if not given `map.data` is plotted).
            resolution: Image resolution (typically 4k).
            ticker (int): Axis ticker.
            alpha (float): Figure transparency.
            draw_circle (bool): Draw the solar disk.
            axis_off (bool): Off the axis.

        Returns:
            Method returns None.
        """
        ax = self.__axis__(ticker, axis_off, disk_axis=not axis_off)
        data = data if data is not None else map.data
        norm = map.plot_settings["norm"]
        norm.vmin, norm.vmax = np.percentile(map.data, [30, 99.9])
        ax.imshow(
            data,
            norm=norm,
            cmap="gray",
            origin="lower",
            alpha=alpha,
        )
        if draw_circle:
            self.__circle__(ax, pixel_radius, resolution, color="m")
        return

    def ovearlay_localized_regions(
        self,
        regions: List[dict],
        prob_lower_lim: float = 0,
        add_color_bar: bool = True,
        cmap: str = "Spectral_r",
        ticker: int = None,
        convert_bgc_black: bool = False,
        axis_off: bool = True,
    ) -> None:
        """Overlay the identified CH regions on top of the Disk maps.

        Arguments:
            regions (List[dict]): List of dictonary holding all the information of identified CH regions.
            prob_lower_lim (float): Minimum limit of the color bar.
            add_color_bar (bool): If `true` plot the probability colorbar on right.
            cmap (str): Color map in string - refer `matplotlib` for details.
            ticker (int): Axis ticker.
            convert_bgc_black (bool): Convert BGC color black.
            axis_off (bool): Off the axis.

        Returns:
            Method returns None.
        """
        ticker = ticker if ticker else self.ticker - 1
        ax = self.__axis__(ticker, axis_off, disk_axis=not axis_off)
        keys = list(regions.__dict__.keys())
        limits, probs = (
            np.array([regions.__dict__[key].lim for key in keys]),
            np.array([regions.__dict__[key].prob for key in keys]),
        )
        n_probs = (probs - probs.min()) / (probs.max() - probs.min())
        logger.info(f"Total regions plotted with seperators {len(keys)}")
        norm = matplotlib.colors.Normalize(vmin=prob_lower_lim, vmax=1.0)
        cmap = matplotlib.cm.get_cmap(cmap)
        stacked = np.max(
            [
                regions.__dict__[key].map * p
                for key, p in zip(keys, n_probs)
                if p >= prob_lower_lim
            ],
            axis=0,
        )
        stacked[stacked == 0.0] = np.nan
        im = ax.imshow(stacked, cmap=cmap, norm=norm, origin="lower")
        ax.patch.set_facecolor("black")
        if add_color_bar:
            self.__add_colorbar__(ax, im, label="Probability")
        return
    
    def ovearlay_localized_contours(
        self,
        regions: List[dict],
        prob_lower_lim: float = 0,
        add_color_bar: bool = True,
        cmap: str = "Spectral_r",
        ticker: int = None,
        convert_bgc_black: bool = False,
        resolution: int = 4096,
        axis_off: bool = True,
    ) -> None:
        """Overlay the identified CH regions on top of the Disk maps.

        Arguments:
            regions (List[dict]): List of dictonary holding all the information of identified CH regions.
            prob_lower_lim (float): Minimum limit of the color bar.
            add_color_bar (bool): If `true` plot the probability colorbar on right.
            cmap (str): Color map in string - refer `matplotlib` for details.
            ticker (int): Axis ticker.
            convert_bgc_black (bool): Convert BGC color black.
            axis_off (bool): Off the axis.

        Returns:
            Method returns None.
        """
        ticker = ticker if ticker else self.ticker - 1
        ax = self.__axis__(ticker, axis_off, disk_axis=not axis_off)
        keys = list(regions.__dict__.keys())
        limits, probs = (
            np.array([regions.__dict__[key].lim for key in keys]),
            np.array([regions.__dict__[key].prob for key in keys]),
        )
        n_probs = (probs - probs.min()) / (probs.max() - probs.min())
        logger.info(f"Total regions plotted with seperators {len(keys)}")
        norm = matplotlib.colors.Normalize(vmin=prob_lower_lim, vmax=1.0)
        cmap = matplotlib.cm.get_cmap(cmap)
        img = np.zeros((resolution, resolution))
        for key, p in zip(keys, n_probs):
            contours, hierarchy = (
                regions.__dict__[key].contours,
                regions.__dict__[key].hierarchy
            )
            color = cmap(p)
            for component in zip(contours, hierarchy):
                cc, ch = (component[0], component[1])
                if ch[3] < 0:
                    ax.plot(cc[:,0,0], cc[:,0,1], color=color, lw=0.05)
        ax.patch.set_facecolor("black")
        if add_color_bar:
            cpos = [1.04, 0.1, 0.025, 0.8]
            cax = ax.inset_axes(cpos, transform=ax.transAxes)
            cb  = matplotlib.colorbar.ColorbarBase(ax=cax, cmap=cmap, values=sorted([0, 0.2, 0.4, 0.6, 0.8, 1]))
            cb.set_label("Probability")
        return

    def __add_colorbar__(
        self,
        ax: matplotlib.axes.Axes,
        im: matplotlib.image.AxesImage,
        label: str = "",
        xOff: float = 0,
        yOff: float = 0,
    ) -> None:
        """Add a colorbar to the right of an axis.

        Arguments:
            ax (matplotlib.axes.Axes): Image axis.
            im (matplotlib.image.AxesImage): Image of the axis.
            label (str): Colorbar axis label.
            xOff (float): X-axis offset of the colorbar.
            yOff (float): Y-axis offset of the colorbar.

        Returns:
            Method returns None.
        """
        cpos = [1.04 + xOff, 0.1 + yOff, 0.025, 0.8]
        cax = ax.inset_axes(cpos, transform=ax.transAxes)
        cb = self.fig.colorbar(im, ax=ax, cax=cax)
        cb.set_label(label)
        return
    
    def draw_binary_localized_contours(
        self,
        regions: List[dict],
        pixel_radius: int,
        resolution: int = 4096,
        axis_off: bool = True,
    ):
        """Method to add binary CH region maps.

        Arguments:
            regions (List[dict]): List of dictonary holding all the information of identified CH regions.
            pixel_radius (int): Radious of the solar disk.
            resolution (int): Image resolution (typically 4k).
            axis_off (bool): Off the axis.

        Returns:
            Method returns None.
        """
        keys = list(regions.__dict__.keys())
        limits, probs = (
            np.array([regions.__dict__[key].lim for key in keys]),
            np.array([regions.__dict__[key].prob for key in keys]),
        )
        n_probs = (probs - probs.min()) / (probs.max() - probs.min())
        fig_num = len(self.axes)
        total_num_regions = len(keys)
        logger.info(
            f"Total regions plotted with seperators {len(keys)}, but will be plotted {fig_num}"
        )
        keys = keys[:: int(total_num_regions / fig_num)][: len(self.axes)]
        for key, p in zip(keys, n_probs):
            contours, hierarchy = (
                regions.__dict__[key].contours,
                regions.__dict__[key].hierarchy
            )
            txt = r"$\tau=$%s" % key + "\n" + r"$\theta=%.3f$" % p
            img = np.zeros((resolution, resolution))
            ax = self.__axis__(None, axis_off=axis_off, disk_axis=not axis_off)
            cv2.drawContours(img, contours, -1, (255,255,255), 1)
            ax.imshow(img, cmap="gray", vmax=1, vmin=0, origin="lower")
            ax.text(
                0.05,
                0.9,
                txt,
                ha="left",
                va="center",
                transform=ax.transAxes,
                fontdict={"color": "w"},
            )
            self.__circle__(ax, pixel_radius, resolution)
        return

    def plot_binary_localized_maps(
        self,
        regions: List[dict],
        pixel_radius: int,
        resolution: int = 4096,
        axis_off: bool = True,
    ):
        """Method to add binary CH region maps.

        Arguments:
            regions (List[dict]): List of dictonary holding all the information of identified CH regions.
            pixel_radius (int): Radious of the solar disk.
            resolution (int): Image resolution (typically 4k).
            axis_off (bool): Off the axis.

        Returns:
            Method returns None.
        """
        keys = list(regions.__dict__.keys())
        limits, probs = (
            np.array([regions.__dict__[key].lim for key in keys]),
            np.array([regions.__dict__[key].prob for key in keys]),
        )
        n_probs = (probs - probs.min()) / (probs.max() - probs.min())
        fig_num = len(self.axes)
        total_num_regions = len(keys)
        logger.info(
            f"Total regions plotted with seperators {len(keys)}, but will be plotted {fig_num}"
        )
        keys = keys[:: int(total_num_regions / fig_num)][: len(self.axes)]
        for key, p in zip(keys, n_probs):
            map = regions.__dict__[key].map
            txt = r"$\tau=$%s" % key + "\n" + r"$\theta=%.3f$" % p
            self.plot_binary_localized_map(map, pixel_radius, resolution, None, txt, axis_off=axis_off)
        return

    def plot_binary_localized_map(
        self,
        map: sunpy.map.Map,
        pixel_radius: int,
        resolution: int = 4096,
        ticker: int = None,
        txt: str = None,
        axis_off: bool = True,
    ):
        """Method to add binary CH region map.

        Arguments:
            regions (List[dict]): List of dictonary holding all the information of identified CH regions.
            pixel_radius (int): Radious of the solar disk.
            resolution (int): Image resolution (typically 4k).
            ticker (int): Axis ticker.
            txt (str): Text to add.
            axis_off (bool): Off the axis.

        Returns:
            Method returns None.
        """
        ax = self.__axis__(ticker, axis_off=axis_off, disk_axis=not axis_off)
        ax.imshow(map, cmap="gray", vmax=1, vmin=0, origin="lower")
        if txt:
            ax.text(
                0.05,
                0.9,
                txt,
                ha="left",
                va="center",
                transform=ax.transAxes,
                fontdict={"color": "w"},
            )
        self.__circle__(ax, pixel_radius, resolution)
        return

    def annotate(
        self, annotations: List[Annotation], ticker: int = 0, axis_off: bool = False
    ) -> None:
        """Method to add text annotations.

        Arguments:
            annotations (List[chips.plots.Annotation]): List of annotations.
            ticker (int): Axis number to put all the annotations.
            axis_off (bool): Off the axis.

        Returns:
            Method returns None.
        """
        ax = self.__axis__(ticker, axis_off=axis_off)
        for a in annotations:
            ax.text(
                a.xloc,
                a.yloc,
                a.txt,
                ha=a.ha,
                va=a.va,
                transform=ax.transAxes,
                fontdict=a.fontdict,
                rotation=a.rotation,
            )
        return

    def draw_colored_synoptic_map(
        self,
        map: sunpy.map.Map,
        wavelength: int,
        ticker: int = None,
        alpha: float = 1,
    ) -> None:
        """Plotting colored solar syoptic images in the axis.

        Arguments:
            map (sunpy.map.Map): Sunpy map level dataset to plt solar synoptic maps.
            wavelength (int): Wave length of the disk image [171/193/211].
            ticker (int): Axis ticker.
            alpha (float): Figure transparency (0,1).

        Returns:
            Method returns None.
        """
        ax = self.__axis__()
        norm = map.plot_settings["norm"]
        norm.vmin, norm.vmax = np.nanpercentile(map.data, [1, 99.9])
        ax.imshow(
            map.data,
            norm=norm,
            cmap=cm.cmlist[f"sdoaia{wavelength}"],
            origin="lower",
            alpha=alpha,
        )
        return

    def ovearlay_localized_synoptic_regions(
        self,
        regions: List[dict],
        prob_lower_lim: float = 0,
        add_color_bar: bool = True,
        cmap: str = "Spectral_r",
    ) -> None:
        """Overlay the identified CH regions on top of the synoptic maps.

        Arguments:
            regions (List[dict]): List of dictonary holding all the information of identified CH regions.
            prob_lower_lim (float): Minimum limit of the color bar.
            add_color_bar (bool): If `true` plot the probability colorbar on right.
            cmap (str): Color map in string - refer `matplotlib` for details.

        Returns:
            Method returns None.
        """
        ax = self.__axis__(ticker=self.ticker - 1)
        keys = list(regions.__dict__.keys())
        limits, probs = (
            np.array([regions.__dict__[key].lim for key in keys]),
            np.array([regions.__dict__[key].prob for key in keys]),
        )
        n_probs = (probs - probs.min()) / (probs.max() - probs.min())
        logger.info(f"Total regions plotted with seperators {len(keys)}")
        norm = matplotlib.colors.Normalize(vmin=prob_lower_lim, vmax=1.0)
        cmap = matplotlib.cm.get_cmap(cmap)
        stacked = np.max(
            [
                regions.__dict__[key].map * p
                for key, p in zip(keys, n_probs)
                if p >= prob_lower_lim
            ],
            axis=0,
        )
        stacked[stacked == 0.0] = np.nan
        im = ax.imshow(stacked, cmap=cmap, norm=norm, origin="lower")
        if add_color_bar:
            self.__add_colorbar__(ax, im, label="Probability")
        return

    def plot_binary_localized_synoptic_maps(self, regions: List[dict]):
        """Method to add binary CH region synoptic maps.

        Arguments:
            regions (List[dict]): List of dictonary holding all the information of identified CH regions.

        Returns:
            Method returns None.
        """
        keys = list(regions.__dict__.keys())
        limits, probs = (
            np.array([regions.__dict__[key].lim for key in keys]),
            np.array([regions.__dict__[key].prob for key in keys]),
        )
        n_probs = (probs - probs.min()) / (probs.max() - probs.min())
        fig_num = len(self.axes)
        total_num_regions = len(keys)
        logger.info(
            f"Total regions plotted with seperators {len(keys)}, but will be plotted {fig_num}"
        )
        keys = keys[:: int(total_num_regions / fig_num)][: len(self.axes)]
        for key, p in zip(keys, n_probs):
            ax = self.__axis__()
            map = regions.__dict__[key].map
            ax.imshow(map, cmap="gray", vmax=1, vmin=0, origin="lower")
            txt = r"$\tau=$%s" % key + "\n" + r"$\theta=%.3f$" % p
            ax.text(
                0.05,
                0.9,
                txt,
                ha="left",
                va="center",
                transform=ax.transAxes,
                fontdict={"color": "w"},
            )
        return

    def write_parameter_details(self, text, xloc, yloc):
        """Method to write parameter details on top of the figure.

        Arguments:
            text (str): String text to write.
            xloc (float): Floating point relative text location along x-axis
            yloc (float): Floating point relative text location along y-axis

        Returns:
            Method returns None.
        """
        self.fig.suptitle(text, y=yloc, x=xloc);
        return


class ChipsPlotter(object):
    """An object class that holds the summary plots.

    Attributes:
        disk (chips.fetch.SolarDisk): Solar disk object that holds all information for drawing.
        figsize (Tuple): Figure size (width, height)
        dpi (int): Dots per linear inch.
        nrows (int): Number of axis rows in a figure palete.
        ncols (int): Number of axis colums in a figure palete.
        parameter_details (dict): Parameters containing xloc, yloc, and text body for figure
    """

    def __init__(
        self,
        disk,
        figsize: Tuple = (6, 6),
        dpi: int = 240,
        nrows: int = 2,
        ncols: int = 2,
        parameter_details: dict = {},
    ):
        """Initialization method."""
        self.disk = disk
        self.figsize = figsize
        self.dpi = dpi
        self.nrows = nrows
        self.ncols = ncols
        self.parameter_details = parameter_details
        return

    def create_diagonestics_plots(
        self,
        fname: str = None,
        figsize: Tuple = None,
        dpi: int = None,
        nrows: int = None,
        ncols: int = None,
        prob_lower_lim: float = 0.8,
        solid_fill: bool = True,
        vert: np.array = None,
    ) -> None:
        """Method to create diagonestics plots showing different steps of CHIPS.

        Attributes:
            fname (str): File name to save the image (expected file formats png, bmp, jpg, pdf, etc).
            figsize (Tuple): Figure size (width, height)
            dpi (int): Dots per linear inch.
            nrows (int): Number of axis rows in a figure palete.
            ncols (int): Number of axis colums in a figure palete.
            prob_lower_lim (float): Minimum limit of the color bar.
            solid_fill (bool): Create solid filled graph or contours.
            vert (np.array): Vertices of rectange to zoom in a specific region [lower left, upper right points].

        Returns:
            Method returns None.
        """
        figsize = figsize if figsize else self.figsize
        dpi = dpi if dpi else self.dpi
        nrows = nrows if nrows else self.nrows
        ncols = ncols if ncols else self.ncols
        ip = ImagePalette(figsize, dpi, nrows, ncols, vert=vert)
        ip.draw_colored_disk(
            map=self.disk.normalized,
            pixel_radius=self.disk.pixel_radius,
            resolution=self.disk.resolution,
        )
        ip.draw_colored_disk(
            map=self.disk.normalized,
            pixel_radius=self.disk.pixel_radius,
            resolution=self.disk.resolution,
            data=self.disk.solar_filter.filt_disk,
        )
        ip.draw_colored_disk(
            map=self.disk.normalized,
            pixel_radius=self.disk.pixel_radius,
            resolution=self.disk.resolution,
        )
        if solid_fill:
            ip.ovearlay_localized_regions(
                self.disk.solar_ch_regions, prob_lower_lim=prob_lower_lim
            )
        else:
            ip.ovearlay_localized_contours(
                self.disk.solar_ch_regions, prob_lower_lim=prob_lower_lim
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
        ip.annotate(annotations)
        if len(self.parameter_details.keys()) > 0:
            ip.write_parameter_details(
                self.parameter_details["text"],
                self.parameter_details["xloc"],
                self.parameter_details["yloc"]
            )
        if fname:
            ip.save(fname)
        ip.close()
        return

    def create_output_stack(
        self,
        fname: str = None,
        figsize: Tuple = None,
        dpi: int = None,
        nrows: int = None,
        ncols: int = None,
        solid_fill: bool = True,
        vert: np.array = None,
    ) -> None:
        """Method to create stack plots showing different binary CH regional plots identified by CHIPS.

        Attributes:
            fname (str): File name to save the image (expected file formats png, bmp, jpg, pdf, etc).
            figsize (Tuple): Figure size (width, height)
            dpi (int): Dots per linear inch.
            nrows (int): Number of axis rows in a figure palete.
            ncols (int): Number of axis colums in a figure palete.
            solid_fill (bool): Create solid filled graph or contours.
            vert (np.array): Vertices of rectange to zoom in a specific region [lower left, upper right points].

        Returns:
            Method returns None.
        """
        figsize = figsize if figsize else self.figsize
        dpi = dpi if dpi else self.dpi
        nrows = nrows if nrows else self.nrows
        ncols = ncols if ncols else self.ncols
        ip = ImagePalette(figsize, dpi, nrows, ncols, vert=vert)
        if solid_fill:
            ip.plot_binary_localized_maps(
                self.disk.solar_ch_regions,
                self.disk.pixel_radius,
                self.disk.resolution,
            )
        else:
            ip.draw_binary_localized_contours(
                self.disk.solar_ch_regions,
                self.disk.pixel_radius,
                self.disk.resolution,
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
        ip.annotate(annotations)
        if len(self.parameter_details.keys()) > 0:
            ip.write_parameter_details(
                self.parameter_details["text"],
                self.parameter_details["xloc"],
                self.parameter_details["yloc"]
            )
        if fname:
            ip.save(fname)
        ip.close()
        return


class SynopticChipsPlotter(object):
    """An object class that holds the summary plots for synoptic map analysis.

    Attributes:
        synoptic_map (chips.fetch.SynopticMap): Solar synoptic map object that holds all information for drawing.
        figsize (Tuple): Figure size (width, height)
        dpi (int): Dots per linear inch.
        nrows (int): Number of axis rows in a figure palete.
        ncols (int): Number of axis colums in a figure palete.
    """

    def __init__(
        self,
        synoptic_map,
        figsize: Tuple = (6, 9),
        dpi: int = 240,
        nrows: int = 3,
        ncols: int = 1,
    ):
        """Initialization method."""
        self.synoptic_map = synoptic_map
        self.figsize = figsize
        self.dpi = dpi
        self.nrows = nrows
        self.ncols = ncols
        return

    def create_synoptic_diagonestics_plots(
        self,
        fname: str = None,
        figsize: Tuple = None,
        dpi: int = None,
        nrows: int = None,
        ncols: int = None,
        prob_lower_lim: float = 0.8,
    ) -> None:
        """Method to create diagonestics plots showing different steps of CHIPS for Synoptic maps.

        Attributes:
            fname (str): File name to save the image (expected file formats png, bmp, jpg, pdf, etc).
            figsize (Tuple): Figure size (width, height).
            dpi (int): Dots per linear inch.
            nrows (int): Number of axis rows in a figure palete.
            ncols (int): Number of axis colums in a figure palete.
            prob_lower_lim (float): Minimum limit of the color bar.

        Returns:
            Method returns None.
        """
        figsize = figsize if figsize else self.figsize
        dpi = dpi if dpi else self.dpi
        nrows = nrows if nrows else self.nrows
        ncols = ncols if ncols else self.ncols
        ip = ImagePalette(figsize, dpi, nrows, ncols)
        ip.draw_colored_synoptic_map(
            map=self.synoptic_map.raw,
            wavelength=self.synoptic_map.wavelength,
        )
        ip.draw_colored_synoptic_map(
            map=self.synoptic_map.raw,
            wavelength=self.synoptic_map.wavelength,
        )
        ip.ovearlay_localized_synoptic_regions(
            self.synoptic_map.solar_ch_regions,
        )
        annotations = []
        annotations.append(
            Annotation(
                self.synoptic_map.date.strftime("%Y-%m-%d %H:%M"),
                0.05,
                1.05,
                "left",
                "center",
            )
        )
        annotations.append(
            Annotation(
                r"$\lambda=%d\AA$" % self.synoptic_map.wavelength,
                -0.05,
                0.99,
                "center",
                "top",
                rotation=90,
            )
        )
        ip.annotate(annotations)
        if fname:
            ip.save(fname)
        ip.close()
        return

    def create_synoptic_output_stack(
        self,
        fname: str = None,
        figsize: Tuple = None,
        dpi: int = None,
        nrows: int = None,
        ncols: int = None,
    ) -> None:
        """Method to create stack plots showing different binary CH regional plots identified by CHIPS.

        Attributes:
            fname (str): File name to save the image (expected file formats png, bmp, jpg, pdf, etc).
            figsize (Tuple): Figure size (width, height).
            dpi (int): Dots per linear inch.
            nrows (int): Number of axis rows in a figure palete.
            ncols (int): Number of axis colums in a figure palete.

        Returns:
            Method returns None.
        """
        figsize = figsize if figsize else self.figsize
        dpi = dpi if dpi else self.dpi
        nrows = nrows if nrows else self.nrows
        ncols = ncols if ncols else self.ncols
        ip = ImagePalette(figsize, dpi, nrows, ncols)
        ip.plot_binary_localized_synoptic_maps(self.synoptic_map.solar_ch_regions)
        annotations = []
        annotations.append(
            Annotation(
                self.synoptic_map.date.strftime("%Y-%m-%d %H:%M"),
                0.05,
                1.05,
                "left",
                "center",
            )
        )
        annotations.append(
            Annotation(
                r"$\lambda=%d\AA$" % self.synoptic_map.wavelength,
                -0.05,
                0.99,
                "center",
                "top",
                rotation=90,
            )
        )
        ip.annotate(annotations)
        if fname:
            ip.save(fname)
        ip.close()
        return
