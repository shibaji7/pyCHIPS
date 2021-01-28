"""main.py: Module is used to run the available algorithms"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import argparse
from dateutil import parser as prs

from data_pipeline import fetch_filenames

def run_cv2_module_sdo(_dict_):
    """ Run all cv2 related algorithms """
    _files_, _dirs_ = fetch_filenames(_dict_)    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dn", "--date", default=dt.datetime(2015,3,11), help="Date [2015-3-11]", type=prs.parse)
    parser.add_argument("-r", "--resolution", default=512, help="Resolution of the files [512]", type=int)
    parser.add_argument("-w", "--wavelength", default=193, help="Wavelength of the files [193]", type=int)
    parser.add_argument("-l", "--loc", default="sdo", help="Database [sdo]", type=str)
    parser.add_argument("-a", "--algo", default="cv2.", help="Database [cv2.]", type=str)
    parser.add_argument("-v", "--verbose", action="store_false", help="Increase output verbosity [True]")
    args = parser.parse_args()
    _dict_ = {}
    if args.verbose:
        print("\n Parameter list for Bgc simulation ")
        for k in vars(args).keys():
            print("     " + k + "->" + str(vars(args)[k]))
            _dict_[k] = vars(args)[k]
    if _dict_["loc"] == "sdo" and "cv2" in _dict_["algo"]: run_cv2_module_sdo(_dict_)