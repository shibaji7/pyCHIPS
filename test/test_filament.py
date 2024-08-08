"""
    test_method_flow.py: Module tests possible cases of filaments in CH filter
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

from chips.chips import Chips
from chips.cleanup import CleanFl
from chips.fetch import RegisterAIA


#################################################
# Method runs and stores all possible parameters
#################################################
def set_AIA(date):
    aia = RegisterAIA(
        date,
        [171, 193, 211],
        4096,
        apply_psf=False,
        local_file="sunpy/data/aia_lev1_{wavelength}a_{date_str}*.fits",
    )
    return aia


if __name__ == "__main__":
    test_chips = True
    date = dt.datetime(2016, 10, 31, 12, 18)
    date = dt.datetime(2015, 8, 11, 15, 10)
    if test_chips:
        # Test method on the main CHIPS function
        aia = RegisterAIA(
            date,
            [193],
            4096,
            apply_psf=False,
            local_file="sunpy/data/aia_lev1_{wavelength}a_{date_str}*.fits",
        )
        ch = Chips(aia, medfilt_kernel=51, h_bins=500, run_fl_cleanup=True)
        ch.run_CHIPS()
    else:
        # Test method on the Cleanup method
        aia = set_AIA(date)
        cf = CleanFl(aia)
        cf.create_coronal_hole_candidates()
        cf.produce_summary_plots()
