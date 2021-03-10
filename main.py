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
import pandas as pd

from chips import Chips, RegisterAIA

def to_local(ch, w, f="data/CHIPS-Outputs/"):
    import cv2
    import numpy as np
    folder = f + str(w) + "/"
    imgaefolder = folder + "images/"
    if not os.path.exists(imgaefolder): os.system("mkdir -p " + imgaefolder)
    cv2.imwrite(imgaefolder + ch.filename.replace(ch.extn, "_binmaps" + ch.extn), ch.fcontours*255)
    np.savetxt(folder + ch.filename.replace(ch.extn, "_binmaps.txt"), ch.fcontours, fmt="%i")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dn", "--date", default=dt.datetime(2018,5,30,12), help="Date [2018-05-30T12]", type=prs.parse)
    parser.add_argument("-r", "--resolution", default=1024, help="Resolution of the files [1024]", type=int)
    parser.add_argument("-w", "--wavelength", default=193, help="Wavelength of the files [193]", type=int)
    parser.add_argument("-wd", "--write_id", action="store_false", help="Increase output verbosity [True]")
    parser.add_argument("-n", "--num", default=-1, help="Run specific dates stored into .csv file", type=int)
    parser.add_argument("-it", "--intensity_threshold", default=48, help="Default intensity threshold for 193A image", type=int)
    parser.add_argument("-vm", "--vmin", default=10, help="Default intensity threshold for register 193A image", type=int)
    args = parser.parse_args()
    _dict_ = {}
    if args.num >= 0:
        _x = pd.read_csv("data/config/dates.csv", parse_dates=["date"])
        if len(_x) <= args.num: raise Exception("CSV file has length:%d, you entered:%d"%(len(_x),args.num))
        args.date = _x.iloc[args.num]["date"]
    print("\n Parameter list ")
    for k in vars(args).keys():
        print("     " + k + "->" + str(vars(args)[k]))
        _dict_[k] = vars(args)[k]
    #_dict_["prims"] = [10, 1380]
    aia = RegisterAIA(_dict_["date"], _dict_["wavelength"], _dict_["resolution"], vmin=args.vmin)
    ch = Chips(aia.fname, aia.folder, _dict_)
    ch.run()
    to_local(ch, args.wavelength)
    pass