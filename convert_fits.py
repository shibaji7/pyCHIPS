"""convert_fits.py: Module is used to convert fits files to 1024 X 1024 images"""

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
import datetime as dt
import os
import cv2
from astropy.io import fits
from to_remote import get_session
import pandas as pd
from data_pipeline import fetch_filenames
import numpy as np
from PCHI import PCHI

# def fetch_all_filenames(wavelength):
#     """ Fetch file name and dirctory """
#     folder = "data/FITS-Database/{:03d}/".format(wavelength)
#     if not os.path.exists(folder): os.system("mkdir -p " + folder)
#     remote = "data/SDO-AIA-Examples/{:03d}/*.image_lev1.fits".format(wavelength)
#     conn = get_session()
#     _, stdout, _ = conn.ssh.exec_command("ls LFS/LFS_iSWAT/" + remote)
#     _files_ = stdout.read().decode("utf-8").split("\n")[:-1]
#     files = [f.split("/")[-1] for f in _files_]
#     return files, folder, "/".join(remote.split("/")[:-1])+"/"

# def to_figure(dat, wavelength=193):
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     plt.style.context("seaborn")
#     from sunpy.cm import cm
#     sdoaia = cm.cmlist.get("sdoaia"+str(wavelength))
#     fig = plt.figure(dpi=256, figsize=(1024/256,1024/256))
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)
#     ax.imshow(dat, aspect='auto', cmap=sdoaia)
#     return fig

# def fetch_convert_all_files(wavelength=193):
#     import matplotlib.pyplot as plt
#     files, local, remote = fetch_all_filenames(wavelength)
#     conn = get_session()
#     for f in files:
#         conn.from_remote_to_local(remote+f, local+f)
#         hdul = fits.open(local+f)
#         dat = hdul[1].data
#         dsize = (1024, 1024)
#         dat = cv2.resize(dat, dsize)
#         fig = to_figure(dat)
#         fig.savefig(local+f.replace(".fits", ".jpg"), bbox_inches="tight", dpi=256)
#         plt.close()
#         im = cv2.imread(cv2.samples.findFile(local+f.replace(".fits", ".jpg")), cv2.IMREAD_COLOR)
#         im = im[12:-13,13:-12]
#         cv2.imwrite(local+f.replace(".fits", ".jpg"), im)
#         hdul.close()
#         os.remove(local+f)
#         #break
#     conn.close()
#     return

def fetch_all_filenames(wavelength):
    """ Fetch file name and dirctory """
    folder = "data/FITS-Database/{:03d}/".format(wavelength)
    if not os.path.exists(folder): os.system("mkdir -p " + folder)
    remote = "data/SDO-AIA-Examples/{:03d}/*.image_lev1.fits".format(wavelength)
    conn = get_session()
    _, stdout, _ = conn.ssh.exec_command("ls LFS/LFS_iSWAT/" + remote)
    _files_ = stdout.read().decode("utf-8").split("\n")[:-1]
    files = [f.split("/")[-1] for f in _files_]
    return files, folder, "/".join(remote.split("/")[:-1])+"/"

def run_pchi(d, th=128):
    files, folder = fetch_filenames(d, 1024, 193)
    dates = [dt.datetime.strptime(f.split("_")[0]+f.split("_")[1], "%Y%m%d%H%M%S") for f in files]
    x = pd.DataFrame(np.array([dates, files]).T, columns=["date", "fname"])
    x["delt"] = np.abs([(u-d).total_seconds() for u in x.date])
    i = x.delt.idxmin() 
    x = x.iloc[[i]]
    file = x.fname.tolist()[0]
    _dict_ = {"date":d, "resolution": 1024, "wavelength":193, "loc": "sdo", "verbose": True, "draw": True, 
              "write_id":True, "intensity_threshold": th}
    pchi = PCHI(file, folder, _dict_)
    pchi.run()
    pchi.close()
    return

def fetch_convert_all_files(K, wavelength=193):
    files, local, remote = fetch_all_filenames(wavelength)
    ith = [48, 56, 48, 48]
    for j, f in enumerate(files):
        if j==K:
            d = dt.datetime.strptime(f[14:33], "%Y-%m-%dT%H_%M_%S")
            cmd = "python data_pipeline.py -dn %s"%d.strftime("%Y-%m-%d")
            print(cmd)
            #os.system(cmd)
            run_pchi(d, ith[j])
    return

if __name__ == "__main__":
    fetch_convert_all_files(int(sys.argv[1]))