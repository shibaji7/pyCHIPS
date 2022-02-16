#!/usr/bin/env python

"""simulate.py: module is dedicated to produce output from CHIPS analysis solution."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"


import datetime as dt
import argparse
from argparse import Namespace
from dateutil import parser as prs
from loguru import logger
import json

import sys
sys.path.append("py/")
from chips_fits import CHIPS
from chips_fits_synoptic import Synoptic

def setup(args, local="py/config/config.json"):
    """
    Load local parameters
    """
    with open(local, "r") as f: o = json.loads("\n".join(f.readlines()))
    for k in o.keys():
        if (not hasattr(args, k) or vars(args)[k] == None)\
                and isinstance(o[k], dict): setattr(args, k, Namespace(**o[k]))
        elif not hasattr(args, k) or vars(args)[k] == None: setattr(args, k, o[k])
    return args

# Script run can also be done via main program
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-r", "--resolution", default=4096, help="Resolution of the image")
    p.add_argument("-w", "--wavelength", default=193, help="AIA Image waveband")
    p.add_argument("-d", "--date", default=dt.datetime(2018,5,30,12), help="Date (default 2018-05-30T12)", type=prs.parse)
    p.add_argument("-vm", "--vmin", default=35., help="Minimum value for fits threshold")
    p.add_argument("-xs", "--xsplit", default=None, help="Split image X-dirc for histogram analysis")
    p.add_argument("-ys", "--ysplit", default=None, help="Split image Y-dirc for histogram analysis")
    p.add_argument("-rs", "--rscale", default=0.6, help="Rescale factor from Pix to Arcsec")
    p.add_argument("-mk", "--medfilt_kernel", default=3, help="Median filter Kernel Size")
    p.add_argument("-p", "--apply_psf", action="store_true", help="Apply PSF on FITS data (default False)")
    p.add_argument("-c", "--clear", action="store_true", help="Clear pervious stored files (default False)")
    p.add_argument("-sy", "--synoptic", action="store_true", help="Run Synoptic Map (default False)")
    args = setup(p.parse_args())
    logger.info(f"Simulation run using simulate.__main__")
    if args.synoptic: Synoptic(args)
    else: CHIPS(args)