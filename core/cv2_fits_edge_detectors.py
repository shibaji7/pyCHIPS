"""cv2_edge_detectors.py: Module is used to implement different edge detection tecqniues using CV2"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import datetime as dt
import os
import numpy as np
import pandas as pd
import cv2
import json
from to_remote import get_session

from astropy.io import fits

from data_pipeline import fetch_filenames

class FitsEdgeDetection(object):
    """ Edge detection by Open-CV """
    
    def __init__(self, files, folder, remote, _dict_):
        for p in _dict_.keys():
            setattr(self, p, _dict_[p])
        self._files_ = files
        self.folder = folder
        self.remote = remote
        self.conn = get_session()
        if not os.path.exists(self.folder): os.system("mkdir -p " + self.folder)
        self.image = np.zeros((self.resolution, self.resolution, 3), np.uint8)
        self.setup()
        return
    
    def setup(self):
        """ Load parameters """
        with open("data/config/config_cv2.json", "r") as fp:
            dic = json.load(fp)
            for p in dic[str(self.wavelength)].keys():
                setattr(self, p, dic[str(self.wavelength)][p])
        return
    
    def rescale(self, x, lev = -1, fact=4):
        """ Rescale Binary files """
        y = (x-np.min(x)) / (np.max(x)-np.min(x))
        if lev <= 0: lev = int(len(x)*fact)
        x = (y*lev).astype(np.uint8)
        return x
    
    def imwrite(self, fname, x):
        dsize = (int(x.shape[0]/8), int(x.shape[1]/8))
        x = cv2.resize(x, dsize)
        cv2.imwrite(fname, self.rescale(x))
        return
    
    def load_fits(self, f):
        """ Load FITS file """
        self.hdul = fits.open(f)
        self.hdul.verify("fix")
        self.imwrite(f.replace(".fits", ".jpg"), self.hdul[1].data)
        return
    
    def find(self):
        """ Find contours by CV2 """
        self._files_.sort()
        fname = self._files_[0]
        f, r = self.folder + fname, self.remote + "/" + fname
        print(" Proc File:",f)
        if self.conn.chek_remote_file_exists(r): 
            self.conn.from_remote_to_local(r, f)
            self.load_fits(f)
            self.detect_hough_circles(f)
            self.detect_edges(f)
        return self
    
    def close(self):
        self.conn.close()
        self.hdul.close()
        return
    
    def detect_hough_circles(self, f):
        """ Detecting the Hough Circles """
        return
    
    def distance(self, contour):
        M = cv2.moments(contour)
        center = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        d = np.sqrt((self.center[0]-center[0])**2+(self.center[1]-center[1])**2)
        return d
    
    def detect_edges(self, f):
        """ Detect edges in the images """
        return