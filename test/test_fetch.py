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
import unittest
import sys
sys.path.append("../chips/")
from fetch import SolarDisk, RegisterAIA

class TestSolarDisk(unittest.TestCase):

    def test_fetch_191_solar_disk(self):
        aia = SolarDisk(
            dt.datetime(2018, 5, 30, 12),
            191
        )
        self.assertTrue(hasattr(aia, "raw"))
        self.assertIsNotNone(aia.raw)
        self.assertTrue(hasattr(aia, "registred"))
        self.assertIsNotNone(aia.registred)
        self.assertTrue(hasattr(aia, "normalized"))
        self.assertIsNotNone(aia.normalized)

        aia = SolarDisk(
            dt.datetime(2018, 5, 30, 12),
            191, norm=False
        )
        self.assertTrue(hasattr(aia, "raw"))
        self.assertIsNotNone(aia.raw)
        self.assertFalse(hasattr(aia, "normalized"))
        self.assertFalse(hasattr(aia, "registred"))
        return

if __name__ == "__main__":
    unittest.main()