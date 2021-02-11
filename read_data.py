"""read-data.py: Module is used to fetch the images from the SDO data store"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, Shibaji"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"


import os
import datetime as dt
import argparse
from dateutil import parser as prs
from lxml import html
import requests
import glob
import numpy as np

class SDOFile(object):
    """ Class that holds SDO file objects """
    
    def __init__(self, _dict_):
        for p in _dict_.keys():
            setattr(self, p, _dict_[p])
        self.uri = "https://sdo.gsfc.nasa.gov/assets/img/browse/"
        self._fetch_file_list_()
        self.folder = "data/SDO-Database/{:4d}.{:02d}.{:02d}/{:d}/{:04d}/".format(self.date.year, self.date.month,
                                                                                  self.date.day, self.resolution, 
                                                                                  self.wavelength)
        if not os.path.exists("data/SDO-Database/"): os.system("mkdir -p data/SDO-Database/")
        if not os.path.exists(self.folder): os.system("mkdir -p " + self.folder)
        return
    
    def _fetch_file_list_(self):
        uri = self.uri + "index.php?b={:4d}%2F{:02d}%2F{:02d}".format(self.date.year,                                                       
                                                                      self.date.month,
                                                                      self.date.day)
        print(" URI:", uri)
        page = requests.get(uri)
        tree = html.fromstring(page.content)
        self.filenames = tree.xpath("//a[@class=\"name file\"]/text()")
        self.hrefs = []
        for a in tree.xpath("//a[@class=\"name file\"]"):
            items = a.items()
            for item in items:
                if item[0] == "href": self.hrefs.append(self.uri + item[1])
        return
    
    def fetch(self):
        tag = "{:d}_{:04d}.jpg".format(self.resolution, self.wavelength)
        for href, fname in zip(self.hrefs, self.filenames):
            if tag in href: self._download_sdo_data_(href, fname)
        return self
    
    def _download_sdo_data_(self, h, fname):
        print(" Downloading from:", h)
        r = requests.get(h)
        with open(self.folder + fname,"wb") as f: f.write(r.content)
        return

def fetch_sdo(_dict_):
    """ Parse SDO files from remote """
    sdo = SDOFile(_dict_)
    sdo.fetch()
    return

def get_files(dates=[dt.datetime(2018,5,30)], resolution=1024, wavelength=193):
    """ Get data file names and location """
    files = []
    for date in dates:
        folder = "data/SDO-Database/{:4d}.{:02d}.{:02d}/{:d}/{:04d}/".format(date.year, date.month, date.day, resolution, wavelength)
        _fs = [f.split("/")[-1] for f in glob.glob(folder + "*_{:d}_{:04d}.jpg".format(resolution, wavelength))]
        _fdates = [np.abs(dt.datetime.strptime(f.split("_")[0]+f.split("_")[1], "%Y%m%d%H%M%S")-date).total_seconds() for f in _fs]
        _ix = np.argmin(_fdates)
        files.append(folder+_fs[_ix])
    return files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dn", "--date", default=dt.datetime(2018,5,30), help="Date [2015-3-11]", type=prs.parse)
    parser.add_argument("-r", "--resolution", default=1024, help="Resolution of the files [512]", type=int)
    parser.add_argument("-w", "--wavelength", default=193, help="Wavelength of the files [193]", type=int)
    args = parser.parse_args()
    _dict_ = {}
    print("\n Parameter list ")
    for k in vars(args).keys():
        print("     " + k + "->" + str(vars(args)[k]))
        _dict_[k] = vars(args)[k]
    fetch_sdo(_dict_)
    pass