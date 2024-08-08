"""
    test_fetch.py: Module is used to test fetch the data from standford system.
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

from fetch import SolarDisk


class TestSolarDisk(unittest.TestCase):
    def test_fetch_193_solar_disk_norm(self):
        aia = SolarDisk(dt.datetime(2018, 5, 18, 12), 193, apply_psf=True)
        self.assertTrue(hasattr(aia, "raw"))
        self.assertIsNotNone(aia.raw)
        self.assertTrue(hasattr(aia, "normalized"))
        self.assertIsNotNone(aia.normalized)
        return

    def test_fetch_193_solar_disk(self):
        aia = SolarDisk(dt.datetime(2018, 5, 18, 12), 193, norm=False)
        self.assertTrue(hasattr(aia, "raw"))
        self.assertIsNotNone(aia.raw)
        self.assertFalse(hasattr(aia, "normalized"))
        return


if __name__ == "__main__":
    unittest.main()
