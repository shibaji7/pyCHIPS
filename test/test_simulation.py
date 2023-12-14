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

sys.path.append("../chips/")
from fetch import RegisterAIA

from chips import Chips


class TestCHIPS(unittest.TestCase):
    def test_fetch_193_solar_disk_(self):
        aia = RegisterAIA(dt.datetime(2018, 5, 30, 12), [193], [4096])
        chips = Chips(aia)
        chips.run_CHIPS()
        return


if __name__ == "__main__":
    unittest.main()
