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
import sys
import unittest

sys.path.append("../chips/")
from fetch import RegisterAIA
from chips import Chips
from plots import ImagePalette

class TestCHIPSPaper(unittest.TestCase):
    def test_run_analysis(self):
        date = dt.datetime(2018, 5, 30, 12)
        aia171, aia193, aia211 = (
            RegisterAIA(
                date, [171], [4096], 
                apply_psf=False
            ),
            RegisterAIA(
                date, [193], [4096], 
                apply_psf=False
            ),
            RegisterAIA(
                date, [211], [4096], 
                apply_psf=False
            )
        )
        chips171, chips193, chips211 = (
            Chips(aia171, medfilt_kernel=11),
            Chips(aia193, medfilt_kernel=11, threshold_range=[-3, 25]),
            Chips(aia211, medfilt_kernel=11, threshold_range=[-3, 5]),
        )
        chips171.run_CHIPS()
        chips193.run_CHIPS()
        chips211.run_CHIPS()
        return

    def create_stack_plot_(self, chips171, chips193, chips211):
        ip = ImagePalette(
            figsize = (9, 3),
            dpi = 300,
            nrows = 1,
            ncols = 3,
        )
        disk171 = chips171.aia.datasets[171][4096]
        ip.draw_colored_disk(
            map=disk171.normalized,
            pixel_radius=disk171.pixel_radius,
            resolution=disk171.resolution,
        )
        ip.save(f"tmp/Figure01.png")
        ip.close()
        return


if __name__ == "__main__":
    unittest.main()
