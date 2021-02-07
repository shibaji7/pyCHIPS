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

from data_pipeline import fetch_filenames, fetch_fits_filename
from cv2_edge_detectors import EdgeDetection
from PCHI import PCHI

def run_cv2_module_sdo(_dict_):
    """ Run all cv2 related algorithms """
    if _dict_["loc"] == "sdo":
        _files_, _dir_ = fetch_filenames(_dict_["date"], _dict_["resolution"], _dict_["wavelength"])
        pchi = PCHI(_files_[0], _dir_, _dict_)
        pchi.run()
        #ed = EdgeDetection(_files_, _dir_, _dict_).find()
        #ed.close()
        pchi.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dn", "--date", default=dt.datetime(2018,5,30,12), help="Date [2018,5,30,12]", type=prs.parse)
    parser.add_argument("-r", "--resolution", default=1024, help="Resolution of the files [1024]", type=int)
    parser.add_argument("-w", "--wavelength", default=193, help="Wavelength of the files [193]", type=int)
    parser.add_argument("-l", "--loc", default="sdo", help="Database [sdo]", type=str)
    parser.add_argument("-v", "--verbose", action="store_false", help="Increase output verbosity [True]")
    parser.add_argument("-wd", "--write_id", action="store_false", help="Increase output verbosity [True]")
    parser.add_argument("-dr", "--draw", action="store_true", help="Increase output verbosity [False]")
    args = parser.parse_args()
    _dict_ = {}
    if args.verbose:
        print("\n Parameter list for Bgc simulation ")
        for k in vars(args).keys():
            print("     " + k + "->" + str(vars(args)[k]))
            _dict_[k] = vars(args)[k]
    run_cv2_module_sdo(_dict_)