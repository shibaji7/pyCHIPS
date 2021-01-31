"""main.py: Module is used to run the available algorithms"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import sys
sys.path.append("core/")
import argparse
import datetime as dt
from dateutil import parser as prs

from cv2_edge_detectors import CHBDetection

def run_cv2_module_sdo(_dict_):
    """ Run all cv2 related algorithms """
    chb = CHBDetection(_dict_).find()
    chb.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dn", "--date", default=dt.datetime(2015,3,11), help="Date [2015-3-11]", type=prs.parse)
    parser.add_argument("-r", "--resolution", default=512, help="Resolution of the files [512]", type=int)
    parser.add_argument("-w", "--lead_wavelength", default=193, help="Wavelength of the files [193]", type=int)
    parser.add_argument("-v", "--verbose", action="store_false", help="Increase output verbosity [True]")
    parser.add_argument("-dr", "--draw", action="store_true", help="Increase output verbosity [False]")
    args = parser.parse_args()
    _dict_ = {}
    if args.verbose:
        print("\n Parameter list for Bgc simulation ")
        for k in vars(args).keys():
            print("     " + k + "->" + str(vars(args)[k]))
            _dict_[k] = vars(args)[k]
    run_cv2_module_sdo(_dict_)