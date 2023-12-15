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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


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
    ):
        self.txt = txt
        self.xloc = xloc
        self.yloc = yloc
        self.ha = ha
        self.va = va
        self.rotation = rotation
        self.fontdict = fontdict
        return


class ImagePalette(object):
    """An object class that holds annotation details that put on a disk image.

    Attributes:
        txt (str):
        xloc (float):
        ha (str):
        va (str):
        fontdict (dict):
        rotation (float):
    """

    def __init__(
        self,
        figsize=(6, 6),
        dpi=240,
        nrows=1,
        ncols=1,
        font_family="sans-serif",
    ):
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
            sharex="all",
            sharey="all",
            figsize=figsize,
            dpi=dpi,
        )
        self.ticker = 0
        self.axes = []
        if nrows * ncols == 1:
            ax = self.axes.append(axs)
        else:
            self.axes.extend(axs.ravel())
        return

    def close(self):
        plt.close("all")
        return

    def save(self, fname):
        self.fig.subplots_adjust(hspace=0.01, wspace=0.01)
        self.fig.savefig(fname, bbox_inches="tight")
        return

    def __axis__(self, ticker=None):
        if ticker is not None:
            ax = self.axes[ticker]
        else:
            ax = self.axes[self.ticker]
            self.ticker += 1
        ax.set_axis_off()
        return ax

    def __circle__(self, ax, pixel_radius, resolution):
        ax.add_patch(
            plt.Circle(
                (resolution / 2, resolution / 2), pixel_radius, color="w", fill=False
            )
        )
        return

    def draw_colored_disk(
        self, map, pixel_radius, data=None, resolution=4096, ticker=None, alpha=1
    ):
        ax = self.__axis__(ticker)
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
        self.__circle__(ax, pixel_radius, resolution)
        return

    def ovearlay_localized_regions(self, regions, prob_lower_lim=0.8):
        ax = self.__axis__(ticker=self.ticker - 1)
        keys = list(regions.__dict__.keys())
        limits, probs = (
            np.array([regions.__dict__[key].lim for key in keys]),
            np.array([regions.__dict__[key].prob for key in keys]),
        )
        n_probs = (probs - probs.min()) / (probs.max() - probs.min())
        logger.info(f"Total regions plotted with seperators {len(keys)}")
        norm = matplotlib.colors.Normalize(vmin=prob_lower_lim, vmax=1.0)
        cmap = matplotlib.cm.get_cmap("Spectral_r")
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
        self._add_colorbar(ax, im, label="Probability")
        return

    def _add_colorbar(
        self,
        ax,
        im,
        label="",
        xOff=0,
        yOff=0,
    ):
        """
        Add a colorbar to the right of an axis.
        """
        cpos = [1.04 + xOff, 0.1 + yOff, 0.025, 0.8]
        cax = ax.inset_axes(cpos, transform=ax.transAxes)
        cb = self.fig.colorbar(im, ax=ax, cax=cax)
        cb.set_label(label)
        return

    def plot_binary_localized_maps(
        self,
        regions,
        pixel_radius,
        resolution=4096,
    ):
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
            txt = r"$\tau=$%s" % key + "\n" + r"$\mathcal{p}=%.3f$" % p
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

    def annotate(self, annotations, ticker=0):
        ax = self.__axis__(ticker)
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


class ChipsPlotter(object):
    """ """

    def __init__(
        self,
        disk,
        figsize=(6, 6),
        dpi=240,
        nrows=2,
        ncols=2,
    ):
        self.disk = disk
        self.figsize = figsize
        self.dpi = dpi
        self.nrows = nrows
        self.ncols = ncols
        return

    def create_diagonestics_plots(
        self,
        fname=None,
        figsize=None,
        dpi=None,
        nrows=None,
        ncols=None,
        prob_lower_lim=0.8,
    ):
        figsize = figsize if figsize else self.figsize
        dpi = dpi if dpi else self.dpi
        nrows = nrows if nrows else self.nrows
        ncols = ncols if ncols else self.ncols
        ip = ImagePalette((9, 3), dpi, 1, 3)
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
        ip.ovearlay_localized_regions(
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
        if fname:
            ip.save(fname)
        ip.close()
        return

    def create_output_stack(
        self,
        fname=None,
        figsize=None,
        dpi=None,
        nrows=None,
        ncols=None,
    ):
        figsize = figsize if figsize else self.figsize
        dpi = dpi if dpi else self.dpi
        nrows = nrows if nrows else self.nrows
        ncols = ncols if ncols else self.ncols
        ip = ImagePalette(figsize, dpi, nrows, ncols)
        ip.plot_binary_localized_maps(
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
        if fname:
            ip.save(fname)
        ip.close()
        return
