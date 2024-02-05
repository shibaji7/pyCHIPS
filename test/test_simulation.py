"""
    test_simulations.py: Module is used to test chips simulations.
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

from chips.fetch import RegisterAIA

from chips.chips import Chips


class TestCHIPS(unittest.TestCase):
    def test_fetch_193_solar_disk_(self):
        aia = RegisterAIA(dt.datetime(2018, 5, 30, 12), [193], [4096], apply_psf=False)
        chips = Chips(
            aia,
            medfilt_kernel=11
        )
        chips.run_CHIPS()
        disk = chips.aia.datasets[193][4096]
        histogram = disk.histogram
        return


if __name__ == "__main__":
    unittest.main()
