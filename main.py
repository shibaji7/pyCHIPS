"""main.py: Module is used to fetch the images and run chrips algorithm"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, Shibaji"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import sys
sys.path.append("core/")
import os
import datetime as dt
import argparse
from dateutil import parser as prs

from read_data import get_files
from chrips import Chrips

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dn", "--date", default=dt.datetime(2018,5,30), help="Date [2015-3-11]", type=prs.parse)
    parser.add_argument("-r", "--resolution", default=1024, help="Resolution of the files [512]", type=int)
    parser.add_argument("-w", "--wavelength", default=193, help="Wavelength of the files [193]", type=int)
    parser.add_argument("-wd", "--write_id", action="store_false", help="Increase output verbosity [True]")
    args = parser.parse_args()
    _dict_ = {}
    print("\n Parameter list ")
    for k in vars(args).keys():
        print("     " + k + "->" + str(vars(args)[k]))
        _dict_[k] = vars(args)[k]
    files = get_files([_dict_["date"]], _dict_["resolution"], _dict_["wavelength"])
    for f in files:
        ch = Chrips(f.split("/")[-1], "/".join(f.split("/")[:-1])+"/", _dict_)
        ch.run()
    pass