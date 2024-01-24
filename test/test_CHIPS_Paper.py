"""
    test_CHIPS_Paper.py: Module is used to test fetch the data from standford system.
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
import os
import pickle
import sys
import unittest

import numpy as np
from sklearn import mixture

path_appended = False
if path_appended:
    sys.path.append("../chips/")
    from fetch import RegisterAIA
    from plots import Annotation, ImagePalette

    from chips import Chips
else:
    from chips.chips import Chips
    from chips.fetch import RegisterAIA
    from chips.plots import Annotation, ImagePalette


class TestCHIPSPaper(unittest.TestCase):
    def test_run_analysis(self):
        fname = "tmp/saveobj.pickle"
        if os.path.exists(fname):
            os.remove(fname)
        self.date = dt.datetime(2018, 6, 23, 13)

        if not os.path.exists(fname):
            aia171, aia193, aia211 = (
                RegisterAIA(self.date, [171], [4096], apply_psf=False),
                RegisterAIA(self.date, [193], [4096], apply_psf=False),
                RegisterAIA(self.date, [211], [4096], apply_psf=False),
            )
            chips171, chips193, chips211 = (
                Chips(aia171, medfilt_kernel=11),
                Chips(aia193, medfilt_kernel=11, threshold_range=[-3, 25]),
                Chips(aia211, medfilt_kernel=11, threshold_range=[-3, 5]),
            )
            chips171.run_CHIPS()
            chips193.run_CHIPS()
            chips211.run_CHIPS()
            with open(fname, "wb") as f:
                pickle.dump(
                    dict(chips171=chips171, chips193=chips193, chips211=chips211), f
                )
        else:
            with open(fname, "rb") as f:
                o = pickle.load(f)
            chips171, chips193, chips211 = o["chips171"], o["chips193"], o["chips211"]
            self.create_stack_plot_fig_analysis(chips171, chips193, chips211)
            # self.create_stack_plot_fig_all(chips171, chips193, chips211)
            self.create_process_flow_steps(chips193)
            self.create_histogram_plots(chips193)
        return

    def __helper_fig2__(self, ip, disk):
        regions = disk.solar_ch_regions
        ip.draw_grayscale_disk(
            map=disk.normalized,
            pixel_radius=disk.pixel_radius,
            resolution=disk.resolution,
        )
        keys = list(regions.__dict__.keys())
        limits, probs = (
            np.array([regions.__dict__[key].lim for key in keys]),
            np.array([regions.__dict__[key].prob for key in keys]),
        )
        n_probs = (probs - probs.min()) / (probs.max() - probs.min())
        p, key = n_probs[2], keys[2]
        txt = r"$\tau=$%s" % key + "\n" + r"$\mathcal{p}=%.3f$" % p
        map = regions.__dict__[key].map
        ip.plot_binary_localized_map(map, disk.pixel_radius, disk.resolution, None, txt)
        ip.draw_colored_disk(
            map=disk.normalized,
            pixel_radius=disk.pixel_radius,
            resolution=disk.resolution,
        )
        ip.ovearlay_localized_regions(
            regions, prob_lower_lim=0.0, convert_bgc_black=True
        )
        return

    def create_stack_plot_fig_analysis(self, chips171, chips193, chips211):
        ip = ImagePalette(
            figsize=(9, 9),
            dpi=300,
            nrows=3,
            ncols=3,
        )
        disk171, disk193, disk211 = (
            chips171.aia.datasets[171][4096],
            chips193.aia.datasets[193][4096],
            chips211.aia.datasets[211][4096],
        )
        self.__helper_fig2__(ip, disk171)
        self.__helper_fig2__(ip, disk193)
        self.__helper_fig2__(ip, disk211)
        annotations = []
        annotations.append(
            Annotation(
                r"$\lambda=%d\AA$" % disk171.wavelength,
                -0.05,
                0.99,
                "center",
                "top",
                rotation=90,
            )
        )
        annotations.append(
            Annotation(
                self.date.strftime("%Y-%m-%d %H:%M"),
                0.05,
                1.05,
                "left",
                "center",
            )
        )
        ip.annotate(annotations)
        ip.annotate(
            [
                Annotation(
                    r"$\lambda=%d\AA$" % disk193.wavelength,
                    -0.05,
                    0.99,
                    "center",
                    "top",
                    rotation=90,
                )
            ],
            3,
        )
        ip.annotate(
            [
                Annotation(
                    r"$\lambda=%d\AA$" % disk211.wavelength,
                    -0.05,
                    0.99,
                    "center",
                    "top",
                    rotation=90,
                )
            ],
            6,
        )
        ip.save(f"tmp/final_product.png")
        ip.close()
        return

    def create_stack_plot_fig_all(self, chips171, chips193, chips211):
        ip = ImagePalette(
            figsize=(9, 3),
            dpi=300,
            nrows=1,
            ncols=3,
        )
        disk171, disk193, disk211 = (
            chips171.aia.datasets[171][4096],
            chips193.aia.datasets[193][4096],
            chips211.aia.datasets[211][4096],
        )
        ip.draw_colored_disk(
            map=disk171.normalized,
            pixel_radius=disk171.pixel_radius,
            resolution=disk171.resolution,
        )
        ip.draw_colored_disk(
            map=disk193.normalized,
            pixel_radius=disk193.pixel_radius,
            resolution=disk193.resolution,
        )
        ip.draw_colored_disk(
            map=disk211.normalized,
            pixel_radius=disk211.pixel_radius,
            resolution=disk211.resolution,
        )
        annotations = []
        annotations.append(
            Annotation(
                self.date.strftime("%Y-%m-%d %H:%M"),
                -0.05,
                0.99,
                "center",
                "top",
                rotation=90,
            )
        )
        annotations.append(
            Annotation(
                r"$\lambda=%d\AA$" % disk171.wavelength,
                0.05,
                1.05,
                "left",
                "center",
            )
        )
        ip.annotate(annotations)
        ip.annotate(
            [
                Annotation(
                    r"$\lambda=%d\AA$" % disk193.wavelength,
                    0.05,
                    1.05,
                    "left",
                    "center",
                )
            ],
            1,
        )
        ip.annotate(
            [
                Annotation(
                    r"$\lambda=%d\AA$" % disk211.wavelength,
                    0.05,
                    1.05,
                    "left",
                    "center",
                )
            ],
            2,
        )
        ip.save(f"tmp/all_wv_disk_image.png")
        ip.close()
        return

    def create_histogram_plots(self, chips193):
        """
        Create the histogram plots
        """
        clf = mixture.GaussianMixture(n_components=2, covariance_type="full")
        disk193 = chips193.aia.datasets[193][4096]
        import matplotlib.pyplot as plt

        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [
            "Tahoma",
            "DejaVu Sans",
            "Lucida Grande",
            "Verdana",
        ]
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(3, 3),
            dpi=300,
        )
        h_data = disk193.solar_filter.filt_disk * disk193.solar_mask.n_mask
        h_data = h_data.ravel()[~np.isnan(h_data.ravel())]
        clf.fit(h_data.reshape(-1, 1))
        ax.hist(h_data, bins=2000, histtype="step", color="r", density=True)
        ax.set_xlim(0, 200)
        ax.axvline(26, ls="--", lw=0.7, color="k")
        ax.axvline(97, ls="--", lw=0.7, color="k")
        ax.axvspan(15, 40, alpha=0.1, color="r")
        ax.set_ylabel("Density")
        ax.set_xlabel(r"Intensity ($I_s$)")
        fig.subplots_adjust(hspace=0.01, wspace=0.01)
        fig.savefig("tmp/histogram.png", bbox_inches="tight")
        plt.close("all")

        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(3, 3),
            dpi=300,
        )
        lim_0 = disk193.histogram.peaks[0]
        threshold_range = np.arange(
            chips193.threshold_range[0], chips193.threshold_range[1]
        )
        limits = np.round(lim_0 + threshold_range, 4)
        h_data = h_data[h_data <= limits[5]]
        ps = 1.0 / (1.0 + np.exp(h_data - lim_0).round(4))
        ax.hist(ps, bins=100, histtype="step", color="r", density=True)
        fig.subplots_adjust(hspace=0.01, wspace=0.01)
        fig.savefig("tmp/beta_dist.png", bbox_inches="tight")
        plt.close("all")
        return

    def __create_localized_maps__(self, disk, regions, ip, id):
        keys = list(regions.__dict__.keys())
        limits, probs = (
            np.array([regions.__dict__[key].lim for key in keys]),
            np.array([regions.__dict__[key].prob for key in keys]),
        )
        n_probs = (probs - probs.min()) / (probs.max() - probs.min())
        p, key = n_probs[id], keys[id]
        txt = r"$\tau=$%s" % key + "\n" + r"$\mathcal{p}=%.3f$" % p
        map = regions.__dict__[key].map
        ip.plot_binary_localized_map(map, disk.pixel_radius, disk.resolution, None, txt)
        return

    def create_process_flow_steps(self, chips193):
        """
        Create the histogram plots
        """
        ip = ImagePalette(
            figsize=(9, 6),
            dpi=300,
            nrows=2,
            ncols=3,
            sharex="none",
            sharey="none",
        )
        disk193 = chips193.aia.datasets[193][4096]
        ip.draw_grayscale_disk(
            map=disk193.normalized,
            pixel_radius=disk193.pixel_radius,
            resolution=disk193.resolution,
        )
        ip.draw_grayscale_disk(
            map=disk193.normalized,
            data=disk193.solar_filter.filt_disk,
            pixel_radius=disk193.pixel_radius,
            resolution=disk193.resolution,
        )
        self.__create_localized_maps__(disk193, disk193.solar_ch_regions, ip, 2)
        self.__create_localized_maps__(disk193, disk193.solar_ch_regions, ip, 6)
        self.__create_localized_maps__(disk193, disk193.solar_ch_regions, ip, 18)
        ip.draw_grayscale_disk(
            map=disk193.normalized,
            data=disk193.solar_filter.filt_disk,
            pixel_radius=disk193.pixel_radius,
            resolution=disk193.resolution,
        )
        ip.ovearlay_localized_regions(disk193.solar_ch_regions, prob_lower_lim=0)
        annotations = []
        annotations.append(
            Annotation(
                self.date.strftime("%Y-%m-%d %H:%M"),
                -0.05,
                0.99,
                "center",
                "top",
                rotation=90,
            )
        )
        annotations.append(
            Annotation(
                r"$\lambda=%d\AA$" % disk193.wavelength,
                0.05,
                1.05,
                "left",
                "center",
            )
        )
        annotations.append(
            Annotation(
                "Gray-scale",
                0.95,
                1.05,
                "right",
                "center",
            )
        )
        ip.annotate(annotations)
        ip.annotate(
            [
                Annotation(
                    r"Gaussian Filterd (11$\times$11)",
                    0.95,
                    1.05,
                    "right",
                    "center",
                )
            ],
            ticker=1,
        )
        ip.annotate(
            [
                Annotation(
                    r"BW-Map of CH",
                    0.95,
                    1.05,
                    "right",
                    "center",
                )
            ],
            ticker=2,
        )
        ip.save(f"tmp/flow_steps.png")
        ip.close()
        return


if __name__ == "__main__":
    unittest.main()
