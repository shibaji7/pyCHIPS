"""
    test_proposal.py: Module is used to test chips simulations.
"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import unittest

import numpy as np

from chips.plots import Annotation, ImagePalette


class TestCHIPS(unittest.TestCase):
    # def test_fetch_193_solar_disk_(self):
    #     aia = RegisterAIA(
    #         dt.datetime(2018, 5, 30, 12),
    #         wavelengths=[193],
    #         apply_psf=False
    #     )
    #     chips = Chips(
    #         aia,
    #         medfilt_kernel=51,
    #         h_bins=500
    #     )
    #     chips.run_CHIPS()
    #     return

    def test_run_Hu(self):
        import cv2

        resolution = 4096
        chips = np.loadtxt("tmp/2018-05-30/CHIPS_2018-05-30.txt")
        charm = np.loadtxt("tmp/2018-05-30/CHARM_2018-05-30.txt")
        catch = np.loadtxt("tmp/2018-05-30/CATCH_2018-05-30.txt")
        ip = ImagePalette(
            figsize=(9, 3),
            dpi=300,
            nrows=1,
            ncols=3,
        )
        contours0, hierarchy = cv2.findContours(
            chips.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        img = np.zeros((resolution, resolution))
        cv2.drawContours(img, contours0[8], -1, (255, 255, 255), 1)
        ax = ip.__axis__()
        ax.imshow(
            img,
            vmax=1,
            vmin=0,
            cmap="gray",
        )
        annotations = []
        annotations.append(
            Annotation(
                "2018-05-30 12:00",
                -0.08,
                1.0,
                "left",
                "top",
                rotation=90,
            )
        )
        annotations.append(Annotation("Scheme: CHIPS", 0.05, 1.05, "left", "center"))
        ip.annotate(annotations)
        ip.__circle__(ax, pixel_radius=0.385 * 4096, resolution=4096)

        contours1, hierarchy = cv2.findContours(
            np.flipud(charm).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        img = np.zeros((resolution, resolution))
        cv2.drawContours(img, contours1[4], -1, (255, 255, 255), 1)
        ax = ip.__axis__()
        ax.imshow(
            img,
            vmax=1,
            vmin=0,
            cmap="gray",
        )
        sim = cv2.matchShapes(contours1[4], contours0[8], 1, 0.0)
        annotations = []
        annotations.append(Annotation("Scheme: CHARM", 0.05, 1.05, "left", "center"))
        annotations.append(Annotation("Hu: %.2f" % sim, 0.99, 1.05, "right", "center"))
        ip.annotate(annotations, ticker=1)
        ip.__circle__(ax, pixel_radius=0.385 * 4096, resolution=4096)

        contours2, hierarchy = cv2.findContours(
            np.flipud(catch).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        img = np.zeros((resolution, resolution))
        cv2.drawContours(img, contours2[4], -1, (255, 255, 255), 1)
        ax = ip.__axis__()
        ax.imshow(
            img,
            vmax=1,
            vmin=0,
            cmap="gray",
        )
        sim = cv2.matchShapes(contours2[4], contours0[8], 1, 0.0)
        annotations = []
        annotations.append(Annotation("Scheme: CATCH", 0.05, 1.05, "left", "center"))
        annotations.append(Annotation("Hu: %.2f" % sim, 0.99, 1.05, "right", "center"))
        ip.annotate(annotations, ticker=2)
        ip.__circle__(ax, pixel_radius=0.385 * 4096, resolution=4096)
        ip.save("tmp/Hu.png")
        ip.close()
        return


if __name__ == "__main__":
    unittest.main()
