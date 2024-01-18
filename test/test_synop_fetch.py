"""
    test_synop_fetch.py: Module is used to test fetch the synoptic data from SDO system.
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
from fetch import SynopticMap
from syn_chips import SynopticChips


class TestSolarDisk(unittest.TestCase):
    def test_synoptic_map_193(self):
        syn = SynopticMap(dt.datetime(2018, 5, 30, 12))
        chips = SynopticChips(syn)
        chips.run_CHIPS()
        return


if __name__ == "__main__":
    unittest.main()
